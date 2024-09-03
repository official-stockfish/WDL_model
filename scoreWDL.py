import argparse, json, matplotlib.pyplot as plt, numpy as np, time
from ast import literal_eval
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, minimize


def win_rate(eval: int | np.ndarray, a: float | np.ndarray, b: float | np.ndarray):
    """the win rate in our model is 1 / ( 1 + exp(-(eval - a) / b))"""

    def stable_logistic(z):
        # returns 1 / (1 + exp(-z)) avoiding possible overflows
        if type(z) == np.ndarray:
            mask = z < 0
            res = np.empty_like(z)
            res[mask] = np.exp(z[mask]) / (1.0 + np.exp(z[mask]))
            res[~mask] = 1.0 / (1.0 + np.exp(-z[~mask]))
            return res
        return np.exp(z) / (1.0 + np.exp(z)) if z < 0 else 1.0 / (1.0 + np.exp(-z))

    # guard against unphysical values, treating negative values of b as "zero"
    # during optimizations this will guide b (back) towards positive values
    if type(b) == np.ndarray:
        b = np.maximum(b, 1e-8)
    elif b < 1e-8:
        b = 1e-8

    return stable_logistic((eval - a) / b)


def loss_rate(eval: int | np.ndarray, a: float | np.ndarray, b: float | np.ndarray):
    """our wdl model assumes symmetry in eval"""
    return win_rate(-eval, a, b)


def poly3(x, c_3, c_2, c_1, c_0):
    """compute the value of a polynomial of 3rd order in a point x"""
    return ((c_3 * x + c_2) * x + c_1) * x + c_0


class WdlData:
    """stores wdl raw data counts and wdl densities in six 2D numpy arrays, with
    'coordinates' (mom, eval), for mom = move/material and internal eval"""

    def __init__(self, args):
        self.momType = args.momType
        self.moveMin, self.moveMax = args.moveMin, args.moveMax
        self.materialMin, self.materialMax = args.materialMin, args.materialMax
        self.winMin = args.winMin
        self.NormalizeData = args.NormalizeData
        if self.NormalizeData is not None:
            self.NormalizeData = json.loads(self.NormalizeData)
            self.NormalizeData["as"] = [float(x) for x in self.NormalizeData["as"]]
            self.normalize_to_pawn_value = int(sum(self.NormalizeData["as"]) + 0.5)
            if not "momType" in self.NormalizeData:
                self.NormalizeData["momType"] = "material"
            assert self.NormalizeData["momType"] in [
                "move",
                "material",
            ], "Error: momType must be move or material."
        else:
            self.normalize_to_pawn_value = args.NormalizeToPawnValue

        print(
            "Converting evals with "
            + (
                f"NormalizeToPawnValue = {self.normalize_to_pawn_value}."
                if self.NormalizeData is None
                else f"NormalizeData = {self.NormalizeData}."
            )
        )

        # numpy arrays have nonnegative indices, so save the two offsets for later
        if self.momType == "move":
            dim_mom = self.moveMax - self.moveMin + 1
            self.offset_mom = self.moveMin
        else:
            dim_mom = self.materialMax - self.materialMin + 1
            self.offset_mom = self.materialMin
        self.eval_max = round(args.evalMax * self.normalize_to_pawn_value / 100)
        dim_eval = 2 * self.eval_max + 1
        self.offset_eval = -self.eval_max

        # set up three integer arrays, each counting positions for (mom, eval) leading to win/draw/loss
        # TODO: check if sparse matrices in place of full 2D arrays are faster overall
        self.wins = np.zeros((dim_mom, dim_eval), dtype=int)
        self.draws = np.zeros((dim_mom, dim_eval), dtype=int)
        self.losses = np.zeros((dim_mom, dim_eval), dtype=int)

    def add_to_wdl_counters(self, result, mom, eval, value):
        """add value to the win/draw/loss counter in the appropriate array"""
        mom_idx, eval_idx = mom - self.offset_mom, eval - self.offset_eval
        if result == "W":
            self.wins[mom_idx, eval_idx] += value
        elif result == "D":
            self.draws[mom_idx, eval_idx] += value
        elif result == "L":
            self.losses[mom_idx, eval_idx] += value

    def load_json_data(self, filenames):
        """load the WDL data from json: the keys describe the position (result, move, material, eval),
        and the values are the observed count of these positions"""
        for filename in filenames:
            print(f"Reading eval stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)

                for key, value in data.items() if data else []:
                    result, move, material, eval = literal_eval(key)

                    if move < self.moveMin or move > self.moveMax:
                        continue
                    if material < self.materialMin or material > self.materialMax:
                        continue

                    # convert the cp eval to the internal value by undoing the normalization
                    if self.NormalizeData is None:
                        # undo static rescaling, that was constant in mom
                        a_internal = self.normalize_to_pawn_value
                    else:
                        # undo dynamic rescaling, that was dependent on mom
                        mom = (
                            move
                            if self.NormalizeData["momType"] == "move"
                            else material
                        )
                        mom_clamped = min(
                            max(mom, self.NormalizeData["momMin"]),
                            self.NormalizeData["momMax"],
                        )
                        a_internal = poly3(
                            mom_clamped / self.NormalizeData["momTarget"],
                            *self.NormalizeData["as"],
                        )
                    eval_internal = round(eval * a_internal / 100)

                    if abs(eval_internal) <= self.eval_max:
                        mom = move if self.momType == "move" else material
                        self.add_to_wdl_counters(result, mom, eval_internal, value)

        W, D, L = self.wins.sum(), self.draws.sum(), self.losses.sum()
        print(f"Retained (W,D,L) = ({W}, {D}, {L}) positions.")

        if W + D + L == 0:
            print("No data was found!")
            exit(0)

        # define wdl densities: if total == 0, entries will be NaN (useful for contour plots)
        total = self.wins + self.draws + self.losses
        self.mask = total > 0
        self.w_density = np.full_like(total, np.nan, dtype=float)
        self.d_density = np.full_like(total, np.nan, dtype=float)
        self.l_density = np.full_like(total, np.nan, dtype=float)
        self.w_density[self.mask] = self.wins[self.mask] / total[self.mask]
        self.d_density[self.mask] = self.draws[self.mask] / total[self.mask]
        self.l_density[self.mask] = self.losses[self.mask] / total[self.mask]

    def get_wdl_counts(self, mom):
        """return views of the three 2D raw count arrays for the given value of mom
        (only used within the constructor of ObjectiveFunction)"""
        mom_idx = mom - self.offset_mom  # recover the row index of mom
        eval_mask = self.mask[mom_idx, :]  # find all the evals with wdl data
        evals = np.where(eval_mask)[0] + self.offset_eval  # recover eval values
        w = self.wins[mom_idx, :][eval_mask]
        d = self.draws[mom_idx, :][eval_mask]
        l = self.losses[mom_idx, :][eval_mask]
        return evals, w, d, l

    def get_wdl_densities(self, mom):
        """return views of the three 2D density arrays for the given value of mom"""
        mom_idx = mom - self.offset_mom  # recover the row index of mom
        eval_mask = self.mask[mom_idx, :]  # find all the evals with wdl data
        evals = np.where(eval_mask)[0] + self.offset_eval  # recover eval values
        w = self.w_density[mom_idx, :][eval_mask]
        d = self.d_density[mom_idx, :][eval_mask]
        l = self.l_density[mom_idx, :][eval_mask]
        return evals, w, d, l

    def get_model_data_density(self):
        """only used for legacy contour plots"""
        valid_data = np.where(self.mask)
        xs, ys = valid_data[1] + self.offset_eval, valid_data[0] + self.offset_mom
        zwins = self.w_density[valid_data]
        zdraws = self.d_density[valid_data]
        return xs, ys, zwins, zdraws

    def fit_abs_locally(self, modelFitting):
        """for each value of mom of interest, find a(mom) and b(mom) so that the induced
        1D win rate function best matches the observed win frequencies"""

        # first filter out mom values with too few wins in total
        total_wins = np.sum(self.wins, axis=1)
        mom_mask = total_wins >= self.winMin
        if not np.all(mom_mask):
            print(
                f"Warning: Too little data, so skipping {self.momType} values",
                np.where(~mom_mask)[0] + self.offset_mom,
            )

        # prepare an array for the values of mom for which we will fit a and b
        model_ms = np.where(mom_mask)[0] + self.offset_mom  # will store mom
        model_as = np.empty_like(model_ms, dtype=float)  # will store a(mom)
        model_bs = np.empty_like(model_ms, dtype=float)  # will store b(mom)

        for i in range(len(model_ms)):
            xdata, ywindata, _, _ = self.get_wdl_densities(model_ms[i])

            # find a(mom) and b(mom) via a simple fit of win_rate() to the densities
            popt_ab = self.normalize_to_pawn_value * np.array([1, 1 / 6])
            popt_ab, _ = curve_fit(win_rate, xdata, ywindata, popt_ab)

            # refine the local result based on data, optimizing an objective function
            if modelFitting != "fitDensity":
                # minimize the objective function
                objective_function = ObjectiveFunction(modelFitting, self, model_ms[i])
                popt_ab, _ = objective_function.minimize(popt_ab)

            model_as[i] = popt_ab[0]  # store a(mom)
            model_bs[i] = popt_ab[1]  # store b(mom)

        return model_ms, model_as, model_bs

    def save_distro_plot(self, pngNameDistro):
        total_wins = np.sum(self.wins, axis=1)
        total_draws = np.sum(self.draws, axis=1)
        total_losses = np.sum(self.losses, axis=1)

        index = np.arange(self.wins.shape[0]) + wdl_data.offset_mom

        plt.bar(index, total_wins, label="Wins", color="blue")
        plt.bar(
            index, total_draws, bottom=total_wins, label="Draws", color="lightgreen"
        )
        plt.bar(
            index,
            total_losses,
            bottom=total_wins + total_draws,
            label="Losses",
            color="red",
        )

        plt.xlim(index[0] - 1, index[-1] + 1)
        plt.xticks(
            [index[0]]
            + [t for t in plt.xticks()[0] if index[0] < t < index[-1]]
            + [index[-1]]
        )
        plt.xlabel(self.momType)
        plt.ylabel("Number of Positions")
        plt.title("Distribution of Wins, Draws, and Losses")
        plt.legend()
        plt.savefig(pngNameDistro)
        plt.close()
        print(f"Saved distribution plot to {pngNameDistro}.")


class ObjectiveFunction:
    """provides objective functions that can be minimized to fit the wdl_data"""

    def __init__(
        self,
        modelFitting: str,
        wdl_data: WdlData,
        single_mom: int | None,
        mom_target: int = 0,
    ):
        if modelFitting == "optimizeScore":
            # minimize the l2 error of the predicted score
            self._objective_function = self.scoreError
        elif modelFitting == "optimizeProbability":
            # maximize the likelihood of predicting the game outcome
            self._objective_function = self.evalLogProbability
        else:
            self._objective_function = None
        self.mom_target = mom_target
        self.wins, self.draws, self.losses = [], [], []
        self.total_count = 0
        for mom in (
            np.arange(wdl_data.wins.shape[0]) + wdl_data.offset_mom
            if single_mom is None
            else [single_mom]
        ):
            evals, w, d, l = wdl_data.get_wdl_counts(mom)
            self.total_count += w.sum() + d.sum() + l.sum()
            # keep only nonzero values to speed up objective function evaluations
            # TODO: check if numpy views or sparse matrices instead of zipped lists are faster
            w_mask, d_mask, l_mask = w > 0, d > 0, l > 0
            self.wins.append((mom, list(zip(evals[w_mask], w[w_mask]))))
            self.draws.append((mom, list(zip(evals[d_mask], d[d_mask]))))
            self.losses.append((mom, list(zip(evals[l_mask], l[l_mask]))))

    def get_ab(self, asbs: np.ndarray, mom: int):
        """return p_a(mom), p_b(mom) or a(mom), b(mom) depending on optimization stage"""
        if len(asbs) == 8:
            coeffs_a = asbs[0:4]
            coeffs_b = asbs[4:8]
            a = poly3(mom / self.mom_target, *coeffs_a)
            b = poly3(mom / self.mom_target, *coeffs_b)
        else:
            a = asbs[0]
            b = asbs[1]

        return a, b

    def estimateScore(self, eval: int, a: float, b: float):
        """estimate game score based on probability of WDL"""

        probw = win_rate(eval, a, b)
        probl = loss_rate(eval, a, b)
        probd = 1 - probw - probl
        return probw + 0.5 * probd + 0

    def scoreError(self, asbs: np.ndarray):
        """l2 distance of predicted scores to actual game scores"""
        scoreErr = 0

        for wdl, score in [(self.wins, 1), (self.draws, 0.5), (self.losses, 0)]:
            for mom, zipped in wdl:
                a, b = self.get_ab(asbs, mom)
                for eval, count in zipped:
                    scoreErr += count * (self.estimateScore(eval, a, b) - score) ** 2

        return np.sqrt(scoreErr / self.total_count)

    def evalLogProbability(self, asbs: np.ndarray):
        """-log((product of game outcome probability)**(1/N))"""
        evalLogProb = 0

        for mom, zipped in self.wins:
            a, b = self.get_ab(asbs, mom)
            for eval, count in zipped:
                probw = win_rate(eval, a, b)
                evalLogProb += count * np.log(max(probw, 1e-14))

        for mom, zipped in self.draws:
            a, b = self.get_ab(asbs, mom)
            for eval, count in zipped:
                probw = win_rate(eval, a, b)
                probl = loss_rate(eval, a, b)
                probd = 1 - probw - probl
                evalLogProb += count * np.log(max(probd, 1e-14))

        for mom, zipped in self.losses:
            a, b = self.get_ab(asbs, mom)
            for eval, count in zipped:
                probl = loss_rate(eval, a, b)
                evalLogProb += count * np.log(max(probl, 1e-14))

        return -evalLogProb / self.total_count

    def __call__(self, asbs: np.ndarray):
        return 0 if self._objective_function is None else self._objective_function(asbs)

    def minimize(self, initial_ab: np.ndarray):
        if self._objective_function is None:
            return initial_ab, "No objective function defined, return initial guess."

        res = minimize(
            self._objective_function,
            initial_ab,
            method="Powell",
            options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
        )
        return res.x, res.message


class WdlModel:
    def __init__(self, args):
        self.momTarget = args.momTarget
        self.modelFitting = args.modelFitting

    def wdl_rates(self, eval: np.ndarray, mom: np.ndarray):
        """our wdl model is based on win/loss rate with a and b polynomials in mom,
        where mom = move or material counter"""
        a = poly3(mom / self.momTarget, *self.coeffs_a)
        b = poly3(mom / self.momTarget, *self.coeffs_b)
        w = win_rate(eval, a, b)
        l = loss_rate(eval, a, b)
        return w, 1 - w - l, l

    def poly3_str(self, coeffs: np.ndarray) -> str:
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(val for pair in zip(coeffs, [self.momTarget] * 4) for val in pair)[
                :-1
            ]
        )

    def fit_ab_globally(self, wdl_data: WdlData):
        print(f"Fit WDL model based on {wdl_data.momType}.")

        # for each value of mom of interest, find good fits for a(mom) and b(mom)
        self.ms, self._as, self.bs = wdl_data.fit_abs_locally(self.modelFitting)

        # now capture the functional behavior of a and b as functions of mom,
        # starting with a simple polynomial fit to find p_a and p_b
        self.coeffs_a, _ = curve_fit(poly3, self.ms / self.momTarget, self._as)
        self.coeffs_b, _ = curve_fit(poly3, self.ms / self.momTarget, self.bs)

        # possibly refine p_a and p_b by optimizing a given objective function
        if self.modelFitting != "fitDensity":
            objective_function = ObjectiveFunction(
                self.modelFitting, wdl_data, None, self.momTarget
            )

            popt_all = self.coeffs_a.tolist() + self.coeffs_b.tolist()
            print("Initial objective function: ", objective_function(popt_all))
            popt_all, message = objective_function.minimize(popt_all)
            self.coeffs_a = popt_all[0:4]  # store final p_a
            self.coeffs_b = popt_all[4:8]  # store final p_b
            print("Final objective function:   ", objective_function(popt_all))
            print(message)

        # prepare output
        self.label_p_a = "p_a = " + self.poly3_str(self.coeffs_a)
        self.label_p_b = "p_b = " + self.poly3_str(self.coeffs_b)

        # now we can report the new conversion factor p_a from internal eval to centipawn
        # such that an expected win score of 50% is for an internal eval of p_a(mom)
        # for a static conversion (independent of mom), we provide a constant value
        # NormalizeToPawnValue = round(p_a(yDataTarget)) = round(sum(coeffs_a))
        fsum_a, fsum_b = sum(self.coeffs_a), sum(self.coeffs_b)

        print(f"const int NormalizeToPawnValue = {int(fsum_a + 0.5)};")
        print(f"Corresponding spread = {int(fsum_b + 0.5)};")
        print(f"Corresponding normalized spread = {fsum_b / fsum_a};")
        print(
            f"Draw rate at 0.0 eval at {wdl_data.momType} {self.momTarget} = {1 - 2 / (1 + np.exp(fsum_a / fsum_b))};"
        )

        print("Parameters in internal value units: ")
        print(self.label_p_a + "\n" + self.label_p_b)
        for ab, coeffs in [("a", self.coeffs_a), ("b", self.coeffs_b)]:
            cstr = ", ".join([f"{c:.8f}" for c in coeffs])
            print(f"    constexpr double {ab}s[] = {{{cstr}}};")


class WdlPlot:
    def __init__(self, args):
        self.setting = args.plot
        self.pngName = args.pngName
        self.momPlotMin = args.momPlotMin
        self.momPlotMax = args.momPlotMax

        self.fig, self.axs = plt.subplots(  # set figure size to A4 x 1.5
            2, 3, figsize=(11.69 * 1.5, 8.27 * 1.5), constrained_layout=True
        )
        self.fig.suptitle(
            "Summary of win-draw-loss "
            + ("data" if args.modelFitting == "None" else "model analysis"),
            fontsize="x-large",
        )

    def normalized_axis(self, i: int, j: int, pawn_value: int):
        """provides a second x-axis in pawns, to go with the original axis in internal eval
        if the engine used a dynamic normalization, the labels will only be approximations
        """
        eval_min, eval_max = self.axs[i, j].get_xlim()
        halfpawn_value = pawn_value / 2
        halfpawn_ticks = np.arange(
            eval_min / halfpawn_value, eval_max / halfpawn_value + 1, dtype=int
        )
        ax2 = self.axs[i, j].twiny()
        ax2.set_xticks(halfpawn_ticks * halfpawn_value)  # ticks at full and half pawns
        ax2.set_xticklabels(["" if z % 2 else str(z // 2) for z in halfpawn_ticks])
        ax2.set_xlim(eval_min, eval_max)  # align the data range with original axis

    def sample_wdl_densities(self, wdl_data: WdlData, mom: int):
        """plot wdl sample data at a fixed mom value"""
        xdata, ywindata, ydrawdata, ylossdata = wdl_data.get_wdl_densities(mom)
        self.axs[0, 0].plot(xdata, ywindata, "b.", label="Measured winrate")
        self.axs[0, 0].plot(xdata, ydrawdata, "g.", label="Measured drawrate")
        self.axs[0, 0].plot(xdata, ylossdata, "c.", label="Measured lossrate")

        self.axs[0, 0].set_xlabel(
            "Evaluation [lower: Internal Value units, upper: Pawns]"
        )
        self.axs[0, 0].set_ylabel("outcome")
        self.axs[0, 0].legend(fontsize="small")
        self.axs[0, 0].set_title(f"Measured data at {wdl_data.momType} {mom}")
        # plot between -3 and 3 pawns
        xmax = ((3 * wdl_data.normalize_to_pawn_value) // 100 + 1) * 100
        self.axs[0, 0].set_xlim([-xmax, xmax])

        self.normalized_axis(0, 0, wdl_data.normalize_to_pawn_value)

    def sample_wdl_curves(self, wdl_model: WdlModel, mom: int):
        """add the three wdl model curves to subplot axs[0, 0]"""
        a = poly3(mom / wdl_model.momTarget, *wdl_model.coeffs_a)
        b = poly3(mom / wdl_model.momTarget, *wdl_model.coeffs_b)
        xdata = np.linspace(*self.axs[0, 0].get_xlim(), num=1000)
        winmodel = win_rate(xdata, a, b)
        lossmodel = loss_rate(xdata, a, b)
        self.axs[0, 0].plot(xdata, winmodel, "r-", label="Model")
        self.axs[0, 0].plot(xdata, lossmodel, "r-")
        self.axs[0, 0].plot(xdata, 1 - winmodel - lossmodel, "r-")
        self.axs[0, 0].set_title(
            "Comparison of model and m" + self.axs[0, 0].title.get_text()[1:]
        )

    def poly3_and_contour_plots(self, wdl_data: WdlData, model: WdlModel):
        """plots p_a, p_b against mom, and adds two contour plots each of wdl_data and model"""
        if model is not None:
            self.axs[1, 0].plot(model.ms, model._as, "b.", label="as")
            self.axs[1, 0].plot(
                model.ms,
                poly3(model.ms / model.momTarget, *model.coeffs_a),
                color="red",
                linewidth=2,
                label=model.label_p_a,
            )
            if (
                wdl_data.NormalizeData is not None
                and wdl_data.NormalizeData["momType"] == wdl_data.momType
            ):
                self.axs[1, 0].plot(
                    model.ms,
                    poly3(
                        model.ms / wdl_data.NormalizeData["momTarget"],
                        *wdl_data.NormalizeData["as"],
                    ),
                    color="lightcoral",
                    linestyle="dashed",
                    label="p_a of the input data's model",
                )
            self.axs[1, 0].plot(model.ms, model.bs, "g.", label="bs")
            self.axs[1, 0].plot(
                model.ms,
                poly3(model.ms / model.momTarget, *model.coeffs_b),
                color="magenta",
                label=model.label_p_b,
            )

            self.axs[1, 0].set_xlabel(wdl_data.momType)
            self.axs[1, 0].set_ylabel("parameters (in internal value units)")
            self.axs[1, 0].legend(fontsize="x-small")
            self.axs[1, 0].set_title("Winrate model parameters")
            self.axs[1, 0].set_ylim(bottom=0.0)

        # use legacy way to create contour plots TODO: directly plot 2D arrays
        xs, ys, zwins, zdraws = wdl_data.get_model_data_density()

        # now generate contour plots
        contourlines = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]

        ylabelStr = wdl_data.momType + " (1,3,3,5,9)" * bool(
            wdl_data.momType == "material"
        )
        ymin, ymax = self.momPlotMin, self.momPlotMax
        points = np.array(list(zip(xs, ys)))

        for j, j_str in enumerate(["win", "draw"]):
            # for wins, plot between -1 and 3 pawns, for draws between -2 and 2 pawns
            xmin = -(((1 + j) * wdl_data.normalize_to_pawn_value) // 100 + 1) * 100
            xmax = (((3 - j) * wdl_data.normalize_to_pawn_value) // 100 + 1) * 100
            grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]  # use a 30x22 grid

            for i, i_str in enumerate(
                ["Data", "Model"] if model is not None else ["Data"]
            ):
                self.axs[i, 1 + j].yaxis.grid(True)
                self.axs[i, 1 + j].xaxis.grid(True)
                self.axs[i, 1 + j].set_xlabel(
                    "Evaluation [lower: Internal Value units, upper: Pawns]"
                )
                self.axs[i, 1 + j].set_ylabel(ylabelStr)

                if i_str == "Data":
                    zz = zdraws if j else zwins
                else:
                    zz = model.wdl_rates(xs, ys)[j]
                zz = griddata(points, zz, (grid_x, grid_y))
                cp = self.axs[i, 1 + j].contourf(grid_x, grid_y, zz, contourlines)

                CS = self.axs[i, 1 + j].contour(
                    grid_x, grid_y, zz, contourlines, colors="black"
                )
                self.axs[i, 1 + j].clabel(CS, inline=1, colors="black")
                self.axs[i, 1 + j].set_title(
                    i_str + ": Fraction of positions leading to a " + j_str
                )
                self.normalized_axis(i, 1 + j, wdl_data.normalize_to_pawn_value)

        self.fig.colorbar(cp, ax=self.axs[:, -1], shrink=0.6)
        self.fig.align_labels()
        self.save()

    def save(self):
        plt.savefig(self.pngName, dpi=300)
        if self.setting == "save+show":
            plt.show()
        plt.close()
        print(f"Saved graphics to {self.pngName}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Stockfish's WDL model to fishtest game statistics. "
        + "Given an (internal) evaluation x, the model sets W(x) = 1 / ( 1 + exp(-(x-a)/b)), "
        + "L(x) = W(-x) and D(x) = 1 - W(x) - L(x), where a = p_a(mom) and b = p_b(mom) are "
        + "polynomials in mom (move number or material count). "
        + "The engine can use the polynomial p_a also to compute a 'centipawn' evaluation so "
        + "that 100cp mean W=50%: either x/p_a(mom) (dynamic rescaling) or x/p_a(momTarget) "
        + "(static rescaling). "
        + "To make the calculation of p_a(momTarget) as simple as possible, the script returns "
        + "{c_3, c_2, c_1, c_0} such that p_a(mom) = sum_i c_i (mom/momTarget)^i, and "
        + "analogously for p_b.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help="json file(s) with fishtest games' win/draw/loss statistics",
        default=["scoreWDLstat.json"],
    )
    parser.add_argument(
        "--NormalizeToPawnValue",
        type=int,
        help="The old p_a(momTarget) value possibly needed for converting the games' cp evals to SF's internal eval.",
    )
    parser.add_argument(
        "--NormalizeData",
        type=str,
        help='Dynamic rescaling parameters. E.g. {"momType": "move", "momMin": 11, "momMax": 120, "momTarget": 32, "as": [0.38036525, -2.82015070, 23.17882135, 307.36768407]}.',
    )
    parser.add_argument(
        "--moveMin",
        type=int,
        default=1,
        help="Lower move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--moveMax",
        type=int,
        default=120,
        help="Upper move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--materialMin",
        type=int,
        default=17,
        help="Lower material count limit for filter applied to json data.",
    )
    parser.add_argument(
        "--materialMax",
        type=int,
        default=78,
        help="Upper material count limit for filter applied to json data.",
    )
    parser.add_argument(
        "--evalMax",
        type=int,
        default=400,
        help="Filter for absolute eval (in cp) applied to json data.",
    )
    parser.add_argument(
        "--momType",
        choices=["move", "material"],
        default="material",
        help="Select y-axis data used for plotting and fitting.",
    )
    parser.add_argument(
        "--momTarget",
        type=int,
        default=58,
        help="The polynomials p_a and p_b will be expressed in terms of sum_i c_i (mom/momTarget)^i.",
    )
    parser.add_argument(
        "--modelFitting",
        choices=["fitDensity", "optimizeProbability", "optimizeScore", "None"],
        default="optimizeProbability",
        help="Choice of model fitting: Fit the win rate curves, maximimize the probability of predicting the outcome, minimize the squared error in predicted score, or no fitting.",
    )
    parser.add_argument(
        "--winMin",
        type=int,
        default=10,
        help="Do not fit win rate curves for mom values with fewer wins in the filtered json data.",
    )
    parser.add_argument(
        "--momPlotMin",
        type=int,
        help="Overrides --moveMin/--materialMin for plotting.",
    )
    parser.add_argument(
        "--momPlotMax",
        type=int,
        help="Overrides --moveMax/--materialMax for plotting.",
    )
    parser.add_argument(
        "--momPlotTarget",
        type=int,
        help="Overrides --momTarget for the density subplot.",
    )
    parser.add_argument(
        "--plot",
        choices=["save+show", "save", "no"],
        default="save+show",
        help="Save/show graphics or not. Useful for batch processing.",
    )
    parser.add_argument(
        "--pngName",
        default="scoreWDL.png",
        help="Name of saved graphics file.",
    )
    parser.add_argument(
        "--pngNameDistro",
        help="Name of optional graphics file for raw data distribution plot.",
    )
    args = parser.parse_args()

    if args.NormalizeToPawnValue is None:
        if args.NormalizeData is None:
            args.NormalizeData = '{"momType": "material", "momMin": 17, "momMax": 78, "momTarget": 58, "as": [-37.45051876,121.19101539,-132.78783573,420.70576692]}'
    else:
        assert (
            args.NormalizeData is None
        ), "Error: Can only specify one of --NormalizeToPawnValue and --NormalizeData."

    if args.momPlotMin is None:
        args.momPlotMin = args.moveMin if args.momType == "move" else args.materialMin
    if args.momPlotMax is None:
        args.momPlotMax = args.moveMax if args.momType == "move" else args.materialMax
    if args.momPlotTarget is None:
        args.momPlotTarget = args.momTarget

    tic = time.time()

    wdl_data = WdlData(args)
    wdl_data.load_json_data(args.filename)
    if args.pngNameDistro:
        wdl_data.save_distro_plot(args.pngNameDistro)

    if args.modelFitting != "None":
        wdl_model = WdlModel(args)
        wdl_model.fit_ab_globally(wdl_data)
    else:
        wdl_model = None

    if args.plot != "no":
        print("Preparing plots.")
        wdl_plot = WdlPlot(args)
        wdl_plot.sample_wdl_densities(wdl_data, args.momPlotTarget)
        if wdl_model:
            wdl_plot.sample_wdl_curves(wdl_model, args.momPlotTarget)

        wdl_plot.poly3_and_contour_plots(wdl_data, wdl_model)

    if args.plot != "save+show":
        print(f"Total elapsed time = {time.time() - tic:.2f}s.")
