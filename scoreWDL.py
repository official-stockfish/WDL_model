import argparse, json, matplotlib.pyplot as plt, numpy as np, time
from ast import literal_eval
from collections import Counter
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, minimize
from typing import Literal, Callable, Any


def win_rate(eval: int | np.ndarray, a, b):
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


def loss_rate(eval: int | np.ndarray, a, b):
    """our wdl model assumes symmetry in eval"""
    return win_rate(-eval, a, b)


def poly3(x: float | np.ndarray, c_3, c_2, c_1, c_0) -> float:
    """compute the value of a polynomial of 3rd order in a point x"""
    return ((c_3 * x + c_2) * x + c_1) * x + c_0


def model_wdl_rates(
    eval: int,
    mom: int,
    mom_target: int,
    coeffs_a: list[float],
    coeffs_b: list[float],
) -> tuple[float, float, float]:
    """our wdl model is based on win/loss rate with a and b polynomials in mom,
    where mom = move or material counter"""
    a = poly3(mom / mom_target, *coeffs_a)
    b = poly3(mom / mom_target, *coeffs_b)
    w = win_rate(eval, a, b)
    l = loss_rate(eval, a, b)
    return w, 1 - w - l, l


@dataclass
class ModelDataDensity:
    """Count data converted to densities"""

    xs: list[int]  # internal evals
    ys: list[int]  # mom counters (mom = move or material)
    zwins: list[float]  # corresponding win probabilities
    zdraws: list[float]  # draw probabilities
    zlosses: list[float]  # loss probabilities


class DataLoader:
    def __init__(
        self,
        filenames: list[str],
        NormalizeToPawnValue: int | None,
        NormalizeData: str | None,
    ):
        self.filenames = filenames
        self.NormalizeToPawnValue = NormalizeToPawnValue
        if NormalizeData is not None:
            self.NormalizeData = json.loads(NormalizeData)
            self.NormalizeData["as"] = [float(x) for x in self.NormalizeData["as"]]
        print(
            "Converting evals with "
            + (
                f"NormalizeToPawnValue = {self.NormalizeToPawnValue}."
                if self.NormalizeToPawnValue is not None
                else f"NormalizeData = {self.NormalizeData}."
            )
        )

        # Load the json, in which the key describes the position (result, move, material, eval),
        # and in which the value is the observed count of these positions
        self.inputdata: Counter[str] = Counter()
        for filename in self.filenames:
            print(f"Reading eval stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)
                if not data:
                    data = {}

                for key, value in data.items():
                    self.inputdata[key] += value

    def extract_wdl(
        self,
        moveMin: int,
        moveMax: int,
        yDataFormat: Literal["move", "material"],
    ) -> tuple[
        Counter[tuple[int, int]],
        Counter[tuple[int, int]],
        Counter[tuple[int, int]],
    ]:
        """Extract three arrays, win draw and loss, each counting positions with a given eval (int) and move/material (int)"""
        freq: Counter[tuple[str, int, int, int]] = Counter(
            {literal_eval(k): v for k, v in self.inputdata.items()}
        )

        win: Counter[tuple[int, int]] = Counter()
        draw: Counter[tuple[int, int]] = Counter()
        loss: Counter[tuple[int, int]] = Counter()
        # filter out (eval, yData) WDL data (i.e. material or move summed out)
        for (result, move, material, eval), v in freq.items():
            # exclude large evals and unwanted move numbers
            if abs(eval) > 400 or move < moveMin or move > moveMax:
                continue

            mom = move if yDataFormat == "move" else material

            # convert the cp eval to the internal value by undoing the normalization
            if self.NormalizeToPawnValue is not None:
                # undo static rescaling, that was constant in mom
                a_internal = self.NormalizeToPawnValue
            else:
                # undo dynamic rescaling, that was dependent on mom
                mom_clamped = min(
                    max(mom, self.NormalizeData["yDataMin"]),
                    self.NormalizeData["yDataMax"],
                )
                a_internal = poly3(
                    mom_clamped / self.NormalizeData["yDataTarget"],
                    *self.NormalizeData["as"],
                )
            eval_internal = round(eval * a_internal / 100)

            if result == "W":
                win[eval_internal, mom] += v
            elif result == "D":
                draw[eval_internal, mom] += v
            elif result == "L":
                loss[eval_internal, mom] += v

        print(
            f"Retained (W,D,L) = ({sum(win.values())}, {sum(draw.values())}, {sum(loss.values())}) positions."
        )

        self.normalize_to_pawn_value = (
            self.NormalizeToPawnValue
            if self.NormalizeToPawnValue is not None
            else int(sum(self.NormalizeData["as"]))
        )

        return win, draw, loss

    def get_model_data_density(
        self,
        win: Counter[tuple[int, int]],
        draw: Counter[tuple[int, int]],
        loss: Counter[tuple[int, int]],
    ) -> ModelDataDensity:
        """Turn the counts of positions into densities/frequencies

        x and y will contain all coordinate values for which data is available.
        Here the counts are normalized to frequencies.
        """
        coords = sorted(set(list(win.keys()) + list(draw.keys()) + list(loss.keys())))
        xs, ys, zwins, zdraws, zlosses = [], [], [], [], []
        for x, y in coords:
            xs.append(x)
            ys.append(y)
            total = win[x, y] + draw[x, y] + loss[x, y]
            zwins.append(win[x, y] / total)
            zdraws.append(draw[x, y] / total)
            zlosses.append(loss[x, y] / total)
        return ModelDataDensity(xs, ys, zwins, zdraws, zlosses)


class ObjectiveFunction:
    """Provides objective functions that can be minimized to fit the win draw loss data"""

    def __init__(
        self,
        modelFitting: str,
        win: Counter[tuple[int, int]],
        draw: Counter[tuple[int, int]],
        loss: Counter[tuple[int, int]],
        y_data_target: int,
    ):
        if modelFitting == "optimizeScore":
            # minimize the l2 error of the predicted score
            self.objective_function = self.scoreError
        elif modelFitting == "optimizeProbability":
            # maximize the likelihood of predicting the game outcome
            self.objective_function = self.evalLogProbability
        else:
            self.objective_function = None
        self.win, self.draw, self.loss = win, draw, loss
        self.y_data_target = y_data_target

    def get_ab(self, asbs: list[float], mom: int):
        # returns p_a(mom), p_b(mom) or a(mom), b(mom) depending on optimization stage
        if len(asbs) == 8:
            coeffs_a = asbs[0:4]
            coeffs_b = asbs[4:8]
            a = poly3(mom / self.y_data_target, *coeffs_a)
            b = poly3(mom / self.y_data_target, *coeffs_b)
        else:
            a = asbs[0]
            b = asbs[1]

        return a, b

    def estimateScore(self, asbs: list[float], eval: int, mom: int):
        """Estimate game score based on probability of WDL"""

        a, b = self.get_ab(asbs, mom)
        probw = win_rate(eval, a, b)
        probl = loss_rate(eval, a, b)
        probd = 1 - probw - probl
        return probw + 0.5 * probd + 0

    def scoreError(self, asbs: list[float]):
        """Sum of the squared error on the game score"""
        scoreErr = 0

        for d, score in [
            (self.win.items(), 1),
            (self.draw.items(), 0.5),
            (self.loss.items(), 0),
        ]:
            for (eval, mom), count in d:
                scoreErr += count * (self.estimateScore(asbs, eval, mom) - score) ** 2

        return scoreErr

    def evalLogProbability(self, asbs: list[float]):
        """-log(product of game outcome probability)"""
        evalLogProb = 0

        for (eval, mom), count in self.win.items():
            a, b = self.get_ab(asbs, mom)
            prob = win_rate(eval, a, b)
            evalLogProb += count * np.log(max(prob, 1e-14))

        for (eval, mom), count in self.draw.items():
            a, b = self.get_ab(asbs, mom)
            probw = win_rate(eval, a, b)
            probl = loss_rate(eval, a, b)
            prob = 1 - probw - probl
            evalLogProb += count * np.log(max(prob, 1e-14))

        for (eval, mom), count in self.loss.items():
            a, b = self.get_ab(asbs, mom)
            prob = loss_rate(eval, a, b)
            evalLogProb += count * np.log(max(prob, 1e-14))

        return -evalLogProb

    def minimize(self, initial_ab):
        if self.objective_function is None:
            return initial_ab

        res = minimize(
            self.objective_function,
            initial_ab,
            method="Powell",
            options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
        )
        return res.x, res.message


@dataclass
class ModelData:
    coeffs_a: list[float]
    coeffs_b: list[float]
    model_ms: np.ndarray
    model_as: list[float]
    model_bs: list[float]
    label_as: str
    label_bs: str


class WdlPlot:
    def __init__(self, args, normalize_to_pawn_value: int):
        self.setting = args.plot
        if self.setting == "no":
            return

        self.pgnName = args.pgnName
        self.normalize_to_pawn_value = normalize_to_pawn_value
        self.yData = args.yData
        self.yDataMax = args.yDataMax
        self.yDataTarget = args.yDataTarget
        self.yPlotMin = args.yPlotMin

        self.fig, self.axs = plt.subplots(  # set figure size to A4 x 1.5
            2, 3, figsize=(11.69 * 1.5, 8.27 * 1.5), constrained_layout=True
        )
        self.fig.suptitle(
            "Summary of win-draw-loss "
            + ("data" if args.modelFitting == "None" else "model analysis"),
            fontsize="x-large",
        )

    def normalized_axis(self, i, j):
        """provides a second x-axis in pawns, to go with the original axis in internal eval
        if the engine used a dynamic normalization, the labels will only be exact for
        the old yDataTarget value for mom (move or material counter)"""
        eval_min, eval_max = self.axs[i, j].get_xlim()
        halfpawn_value = self.normalize_to_pawn_value / 2
        halfpawn_ticks = np.arange(
            eval_min / halfpawn_value, eval_max / halfpawn_value + 1, dtype=int
        )
        ax2 = self.axs[i, j].twiny()
        ax2.set_xticks(halfpawn_ticks * halfpawn_value)  # ticks at full and half pawns
        ax2.set_xticklabels(["" if z % 2 else str(z // 2) for z in halfpawn_ticks])
        ax2.set_xlim(eval_min, eval_max)  # align the data range with original axis

    def poly3_str(self, coeffs: list[float], y_data_target: int) -> str:
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(val for pair in zip(coeffs, [y_data_target] * 4) for val in pair)[
                :-1
            ]
        )

    def create_sample_data_y(
        self,
        xdata: np.ndarray,
        y_data_point: int,
        ywindata: list[float],
        ydrawdata: list[float],
        ylossdata: list[float],
    ):
        """plot wdl sample data at a fixed yData point"""
        self.axs[0, 0].plot(xdata, ywindata, "b.", label="Measured winrate")
        self.axs[0, 0].plot(xdata, ydrawdata, "g.", label="Measured drawrate")
        self.axs[0, 0].plot(xdata, ylossdata, "c.", label="Measured lossrate")

        self.axs[0, 0].set_xlabel(
            "Evaluation [lower: Internal Value units, upper: Pawns]"
        )
        self.axs[0, 0].set_ylabel("outcome")
        self.axs[0, 0].legend(fontsize="small")
        self.axs[0, 0].set_title(
            f"Comparison of model and measured data at {self.yData} {y_data_point}"
        )
        # plot between -3 and 3 pawns
        xmax = ((3 * self.normalize_to_pawn_value) // 100 + 1) * 100
        self.axs[0, 0].set_xlim([-xmax, xmax])

        self.normalized_axis(0, 0)

    def create_sample_curve_y(self, a, b):
        """add the three wdl model curves to subplot axs[0, 0]"""
        xdata = np.linspace(*self.axs[0, 0].get_xlim(), num=1000)
        winmodel = win_rate(xdata, a, b)
        lossmodel = loss_rate(xdata, a, b)
        self.axs[0, 0].plot(xdata, winmodel, "r-", label="Model")
        self.axs[0, 0].plot(xdata, lossmodel, "r-")
        self.axs[0, 0].plot(xdata, 1 - winmodel - lossmodel, "r-")

    def create_plots(self, model_data_density: ModelDataDensity, model: ModelData):
        print("Preparing contour plots and plots of model parameters.")

        plot_fitted_model = len(model.model_ms) > 0
        if plot_fitted_model:
            # graphs of a and b as a function of move/material
            self.axs[1, 0].plot(model.model_ms, model.model_as, "b.", label="as")
            self.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / self.yDataTarget, *model.coeffs_a),
                "r-",
                label="fit: " + model.label_as,
            )
            self.axs[1, 0].plot(model.model_ms, model.model_bs, "g.", label="bs")
            self.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / self.yDataTarget, *model.coeffs_b),
                "m-",
                label="fit: " + model.label_bs,
            )

            self.axs[1, 0].set_xlabel(self.yData)
            self.axs[1, 0].set_ylabel("parameters (in internal value units)")
            self.axs[1, 0].legend(fontsize="x-small")
            self.axs[1, 0].set_title("Winrate model parameters")
            self.axs[1, 0].set_ylim(bottom=0.0)

        # now generate contour plots
        contourlines = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]

        ylabelStr = self.yData + " (1,3,3,5,9)" * bool(self.yData == "material")
        ymin, ymax = self.yPlotMin, self.yDataMax
        points = np.array(list(zip(model_data_density.xs, model_data_density.ys)))

        for j, j_str in enumerate(["win", "draw"]):
            # for wins, plot between -1 and 3 pawns, for draws between -2 and 2 pawns
            xmin = -(((1 + j) * self.normalize_to_pawn_value) // 100 + 1) * 100
            xmax = (((3 - j) * self.normalize_to_pawn_value) // 100 + 1) * 100
            grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]  # use a 30x22 grid

            for i, i_str in enumerate(
                ["Data", "Model"] if plot_fitted_model else ["Data"]
            ):
                self.axs[i, 1 + j].yaxis.grid(True)
                self.axs[i, 1 + j].xaxis.grid(True)
                self.axs[i, 1 + j].set_xlabel(
                    "Evaluation [lower: Internal Value units, upper: Pawns]"
                )
                self.axs[i, 1 + j].set_ylabel(ylabelStr)

                if i_str == "Data":
                    zz = model_data_density.zdraws if j else model_data_density.zwins
                else:
                    zz = model_wdl_rates(
                        np.asarray(model_data_density.xs),
                        np.asarray(model_data_density.ys),
                        self.yDataTarget,
                        model.coeffs_a,
                        model.coeffs_b,
                    )[j]
                zz = griddata(points, zz, (grid_x, grid_y))
                cp = self.axs[i, 1 + j].contourf(grid_x, grid_y, zz, contourlines)

                CS = self.axs[i, 1 + j].contour(
                    grid_x, grid_y, zz, contourlines, colors="black"
                )
                self.axs[i, 1 + j].clabel(CS, inline=1, colors="black")
                self.axs[i, 1 + j].set_title(
                    i_str + ": Fraction of positions leading to a " + j_str
                )
                self.normalized_axis(i, 1 + j)

        self.fig.colorbar(cp, ax=self.axs[:, -1], shrink=0.6)
        self.fig.align_labels()
        self.save()

    def save(self):
        plt.savefig(self.pgnName, dpi=300)
        if self.setting == "save+show":
            plt.show()
        plt.close()
        print(f"Saved graphics to {self.pgnName}.")


class WdlModel:
    def __init__(
        self,
        yData: str,
        yDataMin: int,
        yDataMax: int,
        yDataTarget: int,
        modelFitting: Literal[
            "fitDensity", "optimizeProbability", "optimizeScore", "None"
        ],
        normalize_to_pawn_value: int,
        plot: WdlPlot,
    ):
        self.yData = yData
        self.yDataMin = yDataMin
        self.yDataMax = yDataMax
        self.yDataTarget = yDataTarget
        self.modelFitting = modelFitting
        self.normalize_to_pawn_value = normalize_to_pawn_value
        self.plot = plot

    def extract_model_data(
        self,
        xs: list[int],
        ys: list[int],
        zwins: list[float],
        zdraws: list[float],
        zlosses: list[float],
        win: Counter[tuple[int, int]],
        draw: Counter[tuple[int, int]],
        loss: Counter[tuple[int, int]],
    ):
        evals, moms, winrate, drawrate, lossrate = xs, ys, zwins, zdraws, zlosses

        model_ms, model_as, model_bs = [], [], []

        for mom in range(self.yDataMin, self.yDataMax + 1):
            xdata, ywindata, ydrawdata, ylossdata = [], [], [], []
            for i in range(0, len(moms)):
                if not moms[i] == mom:
                    continue
                xdata.append(evals[i])
                ywindata.append(winrate[i])
                ydrawdata.append(drawrate[i])
                ylossdata.append(lossrate[i])

            if len(ywindata) < 10:
                print(
                    f"Warning: Too little data for {self.yData} value {mom}, skip fitting."
                )
                continue

            # get initial values for a(mom) and b(mom) based on a simple fit of the curve
            popt_ab = self.normalize_to_pawn_value * np.array([1, 1 / 6])
            popt_ab, _ = curve_fit(win_rate, xdata, ywindata, popt_ab)

            # refine the local result based on data, optimizing an objective function

            # get the subset of data relevant for this mom
            if self.modelFitting != "fitDensity":
                winsubset: Counter[tuple[int, int]] = Counter()
                drawsubset: Counter[tuple[int, int]] = Counter()
                losssubset: Counter[tuple[int, int]] = Counter()
                for (eval, momkey), count in win.items():
                    if not momkey == mom:
                        continue
                    winsubset[eval, momkey] = count
                for (eval, momkey), count in draw.items():
                    if not momkey == mom:
                        continue
                    drawsubset[eval, momkey] = count
                for (eval, momkey), count in loss.items():
                    if not momkey == mom:
                        continue
                    losssubset[eval, momkey] = count

                # minimize the objective function
                OF = ObjectiveFunction(
                    self.modelFitting,
                    winsubset,
                    drawsubset,
                    losssubset,
                    self.yDataTarget,
                )
                popt_ab, _ = OF.minimize(popt_ab)

            # store result
            model_ms.append(mom)
            model_as.append(popt_ab[0])  # append a(mom)
            model_bs.append(popt_ab[1])  # append b(mom)

            # this shows the observed wdl data for mom=yDataTarget
            if mom == self.yDataTarget and self.plot.setting != "no":
                self.plot.create_sample_data_y(
                    np.asarray(xdata), mom, ywindata, ydrawdata, ylossdata
                )

        return model_as, model_bs, np.asarray(model_ms)

    def fit_model(
        self,
        xs: list[int],
        ys: list[int],
        zwins: list[float],
        zdraws: list[float],
        zlosses: list[float],
        win: Counter[tuple[int, int]],
        draw: Counter[tuple[int, int]],
        loss: Counter[tuple[int, int]],
    ) -> ModelData:
        print(f"Fit WDL model based on {self.yData}.")
        #
        # for each value of mom of interest, find a(mom) and b(mom) so that the induced
        # 1D win rate function best matches the observed win frequencies
        #

        model_as, model_bs, model_ms = self.extract_model_data(
            xs,
            ys,
            zwins,
            zdraws,
            zlosses,
            win,
            draw,
            loss,
        )

        #
        # now capture the functional behavior of a and b as functions of mom
        #

        # start with a simple polynomial fit to find p_a and p_b
        coeffs_a, _ = curve_fit(poly3, model_ms / self.yDataTarget, model_as)
        coeffs_b, _ = curve_fit(poly3, model_ms / self.yDataTarget, model_bs)

        # possibly refine p_a and p_b by optimizing a given objective function
        if self.modelFitting != "fitDensity":
            OF = ObjectiveFunction(self.modelFitting, win, draw, loss, self.yDataTarget)

            popt_all = coeffs_a.tolist() + coeffs_b.tolist()
            print("Initial objective function: ", OF.objective_function(popt_all))
            popt_all, message = OF.minimize(popt_all)
            coeffs_a = popt_all[0:4]  # store final p_a
            coeffs_b = popt_all[4:8]  # store final p_b
            print("Final objective function:   ", OF.objective_function(popt_all))
            print(message)

        # prepare output
        label_as = "as = " + self.plot.poly3_str(coeffs_a, self.yDataTarget)
        label_bs = "bs = " + self.plot.poly3_str(coeffs_b, self.yDataTarget)

        # now we can report the new conversion factor p_a from internal eval to centipawn
        # such that an expected win score of 50% is for an internal eval of p_a(mom)
        # for a static conversion (independent of mom), we provide a constant value
        # NormalizeToPawnValue = int(p_a(yDataTarget)) = int(sum(coeffs_a))
        fsum_a = sum(coeffs_a)
        fsum_b = sum(coeffs_b)

        if self.plot.setting != "no":
            # this shows the fit of the observed wdl data at mom=yDataTarget to
            # the model wdl rates with a=p_a(yDataTarget) and b=p_b(yDataTarget)
            self.plot.create_sample_curve_y(fsum_a, fsum_b)

        print(f"const int NormalizeToPawnValue = {int(fsum_a)};")
        print(f"Corresponding spread = {int(fsum_b)};")
        print(f"Corresponding normalized spread = {fsum_b / fsum_a};")
        print(
            f"Draw rate at 0.0 eval at move {self.yDataTarget} = {1 - 2 / (1 + np.exp(fsum_a / fsum_b))};"
        )

        print("Parameters in internal value units: ")

        # output as and bs
        print(label_as)
        print(label_bs)
        print(
            "     constexpr double as[] = {%13.8f, %13.8f, %13.8f, %13.8f};"
            % tuple(coeffs_a)
        )
        print(
            "     constexpr double bs[] = {%13.8f, %13.8f, %13.8f, %13.8f };"
            % tuple(coeffs_b)
        )

        return ModelData(
            coeffs_a, coeffs_b, model_ms, model_as, model_bs, label_as, label_bs
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fit Stockfish's WDL model to fishtest game statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "filename",
        nargs="*",
        help="json file(s) with fishtest games' WDL statistics",
        default=["scoreWDLstat.json"],
    )
    parser.add_argument(
        "--NormalizeToPawnValue",
        type=int,
        help="Value needed for converting the games' cp evals to the SF's internal eval.",
    )
    parser.add_argument(
        "--NormalizeData",
        type=str,
        help='Allow dynamic conversion. E.g. {"yDataMin": 11, "yDataMax": 120, "yDataTarget": 32, "as": [0.38036525, -2.82015070, 23.17882135, 307.36768407]}.',
    )
    parser.add_argument(
        "--moveMin",
        type=int,
        default=0,
        help="Lower move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--moveMax",
        type=int,
        default=120,
        help="Upper move number limit for filter applied to json data.",
    )
    parser.add_argument(
        "--yData",
        choices=["move", "material"],
        default="move",
        help="Select y-axis data used for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataMin",
        type=int,
        default=3,
        help="Minimum value of yData to consider for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataMax",
        type=int,
        default=120,
        help="Maximum value of yData to consider for plotting and fitting.",
    )
    parser.add_argument(
        "--yDataTarget",
        type=int,
        default=32,
        help="Value of yData at which new rescaled 100cp should correspond to 50:50 winning chances.",
    )
    parser.add_argument(
        "--yPlotMin",
        type=int,
        help="Overrides --yDataMin for plotting.",
    )
    parser.add_argument(
        "--plot",
        choices=["save+show", "save", "no"],
        default="save+show",
        help="Save/show graphics or not. Useful for batch processing.",
    )
    parser.add_argument(
        "--pgnName",
        default="scoreWDL.png",
        help="Name of saved graphics file.",
    )
    parser.add_argument(
        "--modelFitting",
        choices=["fitDensity", "optimizeProbability", "optimizeScore", "None"],
        default="fitDensity",
        help="Choice of model fitting: Fit the win rate curves, maximimize the probability of predicting the outcome, minimize the squared error in predicted score, or no fitting.",
    )

    args = parser.parse_args()

    if args.NormalizeToPawnValue is None:
        if args.NormalizeData is None:
            args.NormalizeToPawnValue = 328
    else:
        assert (
            args.NormalizeData is None
        ), "Error: Can only specify one of --NormalizeToPawnValue and --NormalizeData."

    if args.yData == "material":
        # fix default values for material
        if args.yDataMax == 120 and args.yDataMin == 3:
            args.yDataMin, args.yDataMax = 10, 78

    if args.yPlotMin is None:
        args.yPlotMin = (
            max(10, args.yDataMin) if args.yData == "move" else args.yDataMin
        )

    tic = time.time()

    data_loader = DataLoader(
        args.filename, args.NormalizeToPawnValue, args.NormalizeData
    )

    win, draw, loss = data_loader.extract_wdl(args.moveMin, args.moveMax, args.yData)

    if len(win) + len(draw) + len(loss) == 0:
        print("No data was found!")
        exit(0)

    wdl_plot = WdlPlot(args, data_loader.normalize_to_pawn_value)

    wdl_model = WdlModel(
        args.yData,
        args.yDataMin,
        args.yDataMax,
        args.yDataTarget,
        args.modelFitting,
        data_loader.normalize_to_pawn_value,
        wdl_plot,
    )

    model_data_density = data_loader.get_model_data_density(win, draw, loss)

    model = (
        wdl_model.fit_model(
            model_data_density.xs,
            model_data_density.ys,
            model_data_density.zwins,
            model_data_density.zdraws,
            model_data_density.zlosses,
            win,
            draw,
            loss,
        )
        if args.modelFitting != "None"
        else ModelData([], [], np.asarray([]), [], [], "", "")
    )

    if args.plot != "no":
        wdl_plot.create_plots(model_data_density, model)

    if args.plot != "save+show":
        print(f"Total elapsed time = {time.time() - tic:.2f}s.")
