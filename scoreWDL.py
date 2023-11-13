import argparse, json, matplotlib.pyplot as plt, numpy as np, time
from ast import literal_eval
from collections import Counter
from dataclasses import dataclass
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, minimize
from typing import Literal, Callable, Any


def poly3(x: float | np.ndarray, a, b, c, d) -> float:
    """compute the value of a polynomial of 3rd order in a point x"""
    return ((a * x + b) * x + c) * x + d


class WdlPlot:
    def __init__(self, title: str, pgnName: str):
        self.title = title
        self.pgnName = pgnName

        self.fig, self.axs = plt.subplots(
            2, 3, figsize=(11.69 * 1.5, 8.27 * 1.5), constrained_layout=True
        )

        self.fig.suptitle(self.title, fontsize="x-large")

    def save(self, plot_setting: Literal["save+show", "save", "no"]):
        plt.savefig(self.pgnName, dpi=300)
        if plot_setting == "save+show":
            plt.show()
        plt.close()
        print(f"Saved graphics to {self.pgnName}.")


@dataclass
class ModelDataDensity:
    """Count data converted to densities"""

    xs: list[float]  # evals
    ys: list[int]  # moves or material
    zwins: list[float]  # corresponding win probability
    zdraws: list[float]  # draw prob
    zlosses: list[float]  # loss prob


class DataLoader:
    def __init__(
        self, filenames: list[str], NormalizeToPawnValue: int, NormalizeData: str
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

    def load_json(self) -> Counter[str]:
        """Load the json, in which the key describes the position (result, move, material, eval),
        and in which the value is the observed count of these positions
        """
        inputdata: Counter[str] = Counter()
        for filename in self.filenames:
            print(f"Reading eval stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)

                for key, value in data.items():
                    inputdata[key] += value
        return inputdata

    def extract_wdl(
        self,
        inputdata: Counter[str],
        moveMin: int,
        moveMax: int,
        NormalizeToPawnValue: int,
        NormalizeData: str,
        yDataFormat: Literal["move", "material"],
    ) -> tuple[
        Counter[tuple[float, int]],
        Counter[tuple[float, int]],
        Counter[tuple[float, int]],
    ]:
        """Extract three arrays, win draw and loss, counting positions with a given eval (float) and move/material (int) that are wdl"""
        freq: Counter[tuple[str, int, int, int]] = Counter(
            {literal_eval(k): v for k, v in inputdata.items()}
        )

        win: Counter[tuple[float, int]] = Counter()
        draw: Counter[tuple[float, int]] = Counter()
        loss: Counter[tuple[float, int]] = Counter()
        # filter out (eval, yData) WDL data (i.e. material or move summed out)
        for (result, move, material, eval), v in freq.items():
            # exclude large evals and unwanted move numbers
            if abs(eval) > 400 or move < moveMin or move > moveMax:
                continue

            yData = move if yDataFormat == "move" else material

            # convert the cp eval to the internal value
            if NormalizeToPawnValue is not None:
                eval_internal = eval * NormalizeToPawnValue / 100
            else:
                yDataClamped = min(
                    max(yData, self.NormalizeData["yDataMin"]),
                    self.NormalizeData["yDataMax"],
                )
                a = poly3(
                    yDataClamped / self.NormalizeData["yDataTarget"],
                    *self.NormalizeData["as"],
                )
                eval_internal = eval * a / 100

            if result == "W":
                win[eval_internal, yData] += v
            elif result == "D":
                draw[eval_internal, yData] += v
            elif result == "L":
                loss[eval_internal, yData] += v

        print(
            f"Retained (W,D,L) = ({sum(win.values())}, {sum(draw.values())}, {sum(loss.values())}) positions."
        )

        ntpv = (
            self.NormalizeToPawnValue
            if NormalizeToPawnValue is not None
            else int(sum(self.NormalizeData["as"]))
        )

        return win, draw, loss, ntpv

    def get_model_data_density(
        self,
        win: Counter[tuple[float, int]],
        draw: Counter[tuple[float, int]],
        loss: Counter[tuple[float, int]],
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


#
# model to be fitted in order to predict the winrate from eval and move (or material)
#
# this defines the model functions
#
class ModelFit:
    def __init__(self, y_data_target: int, normalize_to_pawn_value: int):
        self.y_data_target = y_data_target
        self.normalize_to_pawn_value = normalize_to_pawn_value

    @staticmethod
    def win_rate(x: float | np.ndarray, a, b):
        def stable_logistic(z):
            # returns 1 / (1 + exp(-z)) avoiding possible overflows
            return np.where(
                z < 0, np.exp(z) / (1.0 + np.exp(z)), 1.0 / (1.0 + np.exp(-z))
            )

        return stable_logistic((x - a) / b)

    @staticmethod
    def normalized_axis(ax, normalize_to_pawn_value: int):
        ax2 = ax.twiny()
        tickmin = int(np.ceil(ax.get_xlim()[0] / normalize_to_pawn_value)) * 2
        tickmax = int(np.floor(ax.get_xlim()[1] / normalize_to_pawn_value)) * 2 + 1
        new_tick_locations = np.array(
            [x / 2 * normalize_to_pawn_value for x in range(tickmin, tickmax)]
        )

        def tick_function(X):
            V = X / normalize_to_pawn_value
            return [(f"{z:.0f}" if z % 1 < 0.1 else "") for z in V]

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))

    def poly3_str(self, coeffs) -> str:
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(
                val for pair in zip(coeffs, [self.y_data_target] * 4) for val in pair
            )[:-1]
        )

    def wdl(
        self,
        eval: float,
        move_or_material: int,
        popt_as: list[float],
        popt_bs: list[float],
    ) -> tuple[int, int, int]:
        """Compute the integer wdl (per-mille) using polynomial approximation for a and b"""
        a = poly3(move_or_material / self.y_data_target, *popt_as)
        b = poly3(move_or_material / self.y_data_target, *popt_bs)
        w = int(1000 * ModelFit.win_rate(eval, a, b))
        l = int(1000 * ModelFit.win_rate(-eval, a, b))
        d = 1000 - w - l
        return w, d, l


class ObjectiveFunctions:
    """Collects objective functions that can be minimized to fit the win draw loss data"""

    def __init__(self, win, draw, loss, fit):
        self.win = win
        self.draw = draw
        self.loss = loss
        self.fit = fit

    def get_ab(self, asbs: list[float], mom):
        # returns p_a(mom), p_b(mom) or a(mom), b(mom) depending on optimization stage
        if len(asbs) == 8:
            popt_as = asbs[0:4]
            popt_bs = asbs[4:8]
            a = poly3(mom / self.fit.y_data_target, *popt_as)
            b = poly3(mom / self.fit.y_data_target, *popt_bs)
        else:
            a = asbs[0]
            b = asbs[1]

        return a, b

    def estimateScore(self, asbs: list[float], eval, mom):
        """Estimate game score based on probability of WDL"""

        a, b = self.get_ab(asbs, mom)

        # guard against unphysical stuff
        if a <= 0 or b <= 0:
            return 4

        probw = ModelFit.win_rate(eval, a, b)
        probl = ModelFit.win_rate(-eval, a, b)
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
            prob = ModelFit.win_rate(eval, a, b)
            evalLogProb += count * np.log(max(prob, 1e-14))

        for (eval, mom), count in self.draw.items():
            a, b = self.get_ab(asbs, mom)
            probw = ModelFit.win_rate(eval, a, b)
            probl = ModelFit.win_rate(-eval, a, b)
            prob = 1 - (probw + probl)
            evalLogProb += count * np.log(max(prob, 1e-14))

        for (eval, mom), count in self.loss.items():
            a, b = self.get_ab(asbs, mom)
            prob = ModelFit.win_rate(-eval, a, b)
            evalLogProb += count * np.log(max(prob, 1e-14))

        return -evalLogProb


@dataclass
class ModelData:
    popt_as: list[float]
    popt_bs: list[float]
    model_ms: np.ndarray
    model_as: list[float]
    model_bs: list[float]
    label_as: str
    label_bs: str


class WdlModel:
    def __init__(self, args, plot: WdlPlot):
        self.args = args
        self.plot = plot

    def sample_curve_y(
        self,
        xdata: np.ndarray,
        ywindata: list[float],
        ydrawdata: list[float],
        ylossdata: list[float],
        popt_ab,
    ):
        # plot sample curves at yDataTarget
        self.plot.axs[0, 0].plot(xdata, ywindata, "b.", label="Measured winrate")
        self.plot.axs[0, 0].plot(xdata, ydrawdata, "g.", label="Measured drawrate")
        self.plot.axs[0, 0].plot(xdata, ylossdata, "c.", label="Measured lossrate")

        winmodel = ModelFit.win_rate(xdata, popt_ab[0], popt_ab[1])
        lossmodel = ModelFit.win_rate(-xdata, popt_ab[0], popt_ab[1])
        self.plot.axs[0, 0].plot(xdata, winmodel, "r-", label="Model")
        self.plot.axs[0, 0].plot(xdata, lossmodel, "r-")
        self.plot.axs[0, 0].plot(xdata, 1 - winmodel - lossmodel, "r-")

        self.plot.axs[0, 0].set_xlabel(
            "Evaluation [lower: Internal Value units, upper: Pawns]"
        )
        self.plot.axs[0, 0].set_ylabel("outcome")
        self.plot.axs[0, 0].legend(fontsize="small")
        self.plot.axs[0, 0].set_title(
            f"Comparison of model and measured data at {self.args.yData} {self.args.yDataTarget}"
        )
        xmax = ((3 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        self.plot.axs[0, 0].set_xlim([-xmax, xmax])

        ModelFit.normalized_axis(self.plot.axs[0, 0], self.args.NormalizeToPawnValue)

    def extract_model_data(
        self,
        xs: list[float],
        ys: list[int],
        zwins: list[float],
        zdraws: list[float],
        zlosses: list[float],
        win: Counter[tuple[float, int]],
        draw: Counter[tuple[float, int]],
        loss: Counter[tuple[float, int]],
        fit,
        plotfunc: Callable[
            [list[float], list[float], list[float], list[float], Any], None
        ],
    ):
        evals, moms, winrate, drawrate, lossrate = xs, ys, zwins, zdraws, zlosses

        model_ms, model_as, model_bs = [], [], []

        grouping = 1

        # mom = move or material, depending on self.args.yData
        for mom in range(self.args.yDataMin, self.args.yDataMax + 1, grouping):
            mmin, mmax = mom, mom + grouping
            xdata, ywindata, ydrawdata, ylossdata = [], [], [], []
            for i in range(0, len(moms)):
                if moms[i] < mmin or moms[i] >= mmax:
                    continue
                xdata.append(evals[i])
                ywindata.append(winrate[i])
                ydrawdata.append(drawrate[i])
                ylossdata.append(lossrate[i])

            if len(ywindata) < 10:
                print(
                    f"Warning: Too little data for {self.args.yData} value {mom}, skip fitting."
                )
                continue

            popt_ab: tuple[float, float]

            # get initial values for a(mom) and b(mom) based on a simple fit of the curve
            popt_ab, _ = curve_fit(
                ModelFit.win_rate,
                xdata,
                ywindata,
                p0=[self.args.NormalizeToPawnValue, self.args.NormalizeToPawnValue / 6],
            )

            # refine the local result based on data, optimizing an objective function

            # get the subset of data relevant for this mom
            if self.args.modelFitting != "fitDensity":
                winsubset: Counter[tuple[float, int]] = Counter()
                drawsubset: Counter[tuple[float, int]] = Counter()
                losssubset: Counter[tuple[float, int]] = Counter()
                for (eval, momkey), count in win.items():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    winsubset[eval, momkey] = count
                for (eval, momkey), count in draw.items():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    drawsubset[eval, momkey] = count
                for (eval, momkey), count in loss.items():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    losssubset[eval, momkey] = count

                # miniminize the objective function
                OF = ObjectiveFunctions(winsubset, drawsubset, losssubset, fit)

                if self.args.modelFitting == "optimizeScore":
                    objectiveFunction = OF.scoreError
                else:
                    if self.args.modelFitting == "optimizeProbability":
                        objectiveFunction = OF.evalLogProbability
                    else:
                        objectiveFunction = None
                res = minimize(
                    objectiveFunction,
                    popt_ab,
                    method="Powell",
                    options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
                )
                popt_ab = res.x

            # prepare output

            # store result
            model_ms.append(mom)
            model_as.append(popt_ab[0])  # append a(mom)
            model_bs.append(popt_ab[1])  # append b(mom)

            # this shows the "local" fit, using a(yDataTarget) and b(yDataTarget)
            # it probably would be interesting to show p_a and p_b at yDataTarget instead TODO (update Readme.md when done)
            if mom == self.args.yDataTarget and plotfunc != None:
                plotfunc(np.asarray(xdata), ywindata, ydrawdata, ylossdata, popt_ab)

        return model_as, model_bs, np.asarray(model_ms)

    def fit_model(
        self,
        xs: list[float],
        ys: list[int],
        zwins: list[float],
        zdraws: list[float],
        zlosses: list[float],
        win: Counter[tuple[float, int]],
        draw: Counter[tuple[float, int]],
        loss: Counter[tuple[float, int]],
    ) -> ModelData:
        print(f"Fit WDL model based on {self.args.yData}.")
        #
        # for each value of mom of interest, find a(mom) and b(mom) so that the induced
        # 1D win rate function best matches the observed win frequencies
        #

        fit = ModelFit(self.args.yDataTarget, self.args.NormalizeToPawnValue)

        model_as, model_bs, model_ms = self.extract_model_data(
            xs,
            ys,
            zwins,
            zdraws,
            zlosses,
            win,
            draw,
            loss,
            fit,
            self.sample_curve_y if self.args.plot != "no" else None,
        )

        #
        # now capture the functional behavior of a and b as functions of mom
        #

        # simple polynomial fit to find p_a and p_b
        popt_as, _ = curve_fit(poly3, model_ms / fit.y_data_target, model_as)
        popt_bs, _ = curve_fit(poly3, model_ms / fit.y_data_target, model_bs)

        # refinement phase
        #
        # optimize the likelihood of seeing the data ...
        #    our model_as, model_bs / popt_as, popt_bs are just initial guesses
        #

        if self.args.modelFitting != "fitDensity":
            OF = ObjectiveFunctions(win, draw, loss, fit)

            if self.args.modelFitting == "optimizeScore":
                objectiveFunction = OF.scoreError
            else:
                if self.args.modelFitting == "optimizeProbability":
                    objectiveFunction = OF.evalLogProbability
                else:
                    objectiveFunction = None

            popt_all = popt_as.tolist() + popt_bs.tolist()
            print("Initial objective function: ", objectiveFunction(popt_all))
            res = minimize(
                objectiveFunction,
                popt_all,
                method="Powell",
                options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
            )
            popt_as = res.x[0:4]  # store final p_a
            popt_bs = res.x[4:8]  # store final p_b
            popt_all = res.x
            print("Final objective function:   ", objectiveFunction(popt_all))
            print(res.message)

        # prepare output
        label_as, label_bs = "as = " + fit.poly3_str(popt_as), "bs = " + fit.poly3_str(
            popt_bs
        )

        #
        # now we can define the conversion factor from internal eval to centipawn such that
        # an expected win score of 50% is for an eval of 'a', we pick this value for the yDataTarget
        # (where the sum of the a coefs is equal to the interpolated a).
        fsum_a = sum(popt_as)
        fsum_b = sum(popt_bs)
        print(f"const int NormalizeToPawnValue = {int(fsum_a)};")
        print(f"Corresponding spread = {int(fsum_b)};")
        print(f"Corresponding normalized spread = {fsum_b / fsum_a};")
        print(
            f"Draw rate at 0.0 eval at move {self.args.yDataTarget} = {1 - 2 / (1 + np.exp(fsum_a / fsum_b))};"
        )

        print("Parameters in internal value units: ")

        # output as and bs
        print(label_as)
        print(label_bs)
        print(
            "     constexpr double as[] = {%13.8f, %13.8f, %13.8f, %13.8f};"
            % tuple(popt_as)
        )
        print(
            "     constexpr double bs[] = {%13.8f, %13.8f, %13.8f, %13.8f };"
            % tuple(popt_bs)
        )

        return ModelData(
            popt_as, popt_bs, model_ms, model_as, model_bs, label_as, label_bs
        )

    def create_plot(self, model_data_density: ModelDataDensity, model: ModelData):
        # graphs of a and b as a function of move/material
        print("Plotting move/material dependence of model parameters.")

        fit = ModelFit(self.args.yDataTarget, self.args.NormalizeToPawnValue)

        if self.args.modelFitting != "None":
            self.plot.axs[1, 0].plot(model.model_ms, model.model_as, "b.", label="as")
            self.plot.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / fit.y_data_target, *model.popt_as),
                "r-",
                label="fit: " + model.label_as,
            )
            self.plot.axs[1, 0].plot(model.model_ms, model.model_bs, "g.", label="bs")
            self.plot.axs[1, 0].plot(
                model.model_ms,
                poly3(model.model_ms / fit.y_data_target, *model.popt_bs),
                "m-",
                label="fit: " + model.label_bs,
            )

            self.plot.axs[1, 0].set_xlabel(self.args.yData)
            self.plot.axs[1, 0].set_ylabel("parameters (in internal value units)")
            self.plot.axs[1, 0].legend(fontsize="x-small")
            self.plot.axs[1, 0].set_title("Winrate model parameters")
            self.plot.axs[1, 0].set_ylim(bottom=0.0)

        # now generate contour plots
        contourlines = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]

        print("Processing done, plotting 2D data.")
        ylabelStr = self.args.yData + " (1,3,3,5,9)" * bool(
            self.args.yData == "material"
        )
        for i in [0, 1] if self.args.modelFitting != "None" else [0]:
            for j in [1, 2]:
                self.plot.axs[i, j].yaxis.grid(True)
                self.plot.axs[i, j].xaxis.grid(True)
                self.plot.axs[i, j].set_xlabel(
                    "Evaluation [lower: Internal Value units, upper: Pawns]"
                )
                self.plot.axs[i, j].set_ylabel(ylabelStr)

        # for wins, plot between -1 and 3 pawns, using a 30x22 grid
        xmin = -((1 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        xmax = ((3 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        ymin, ymax = self.args.yPlotMin, self.args.yDataMax
        grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]
        points = np.array(list(zip(model_data_density.xs, model_data_density.ys)))

        # data
        zz = griddata(
            points, model_data_density.zwins, (grid_x, grid_y), method="linear"
        )
        cp = self.plot.axs[0, 1].contourf(grid_x, grid_y, zz, contourlines)
        self.plot.fig.colorbar(cp, ax=self.plot.axs[:, -1], shrink=0.618)
        CS = self.plot.axs[0, 1].contour(
            grid_x, grid_y, zz, contourlines, colors="black"
        )
        self.plot.axs[0, 1].clabel(CS, inline=1, colors="black")
        self.plot.axs[0, 1].set_title("Data: Fraction of positions leading to a win")

        ModelFit.normalized_axis(self.plot.axs[0, 1], self.args.NormalizeToPawnValue)

        # model
        if self.args.modelFitting != "None":
            zwins = []
            for i in range(0, len(model_data_density.xs)):
                zwins.append(
                    fit.wdl(
                        model_data_density.xs[i],
                        model_data_density.ys[i],
                        model.popt_as,
                        model.popt_bs,
                    )[0]
                    / 1000.0
                )
            zz = griddata(points, zwins, (grid_x, grid_y), method="linear")
            cp = self.plot.axs[1, 1].contourf(grid_x, grid_y, zz, contourlines)
            CS = self.plot.axs[1, 1].contour(
                grid_x, grid_y, zz, contourlines, colors="black"
            )
            self.plot.axs[1, 1].clabel(CS, inline=1, colors="black")
            self.plot.axs[1, 1].set_title(
                "Model: Fraction of positions leading to a win"
            )
            ModelFit.normalized_axis(
                self.plot.axs[1, 1], self.args.NormalizeToPawnValue
            )

        # for draws, plot between -2 and 2 pawns, using a 30x22 grid
        xmin = -((2 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        xmax = ((2 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]
        points = np.array(list(zip(model_data_density.xs, model_data_density.ys)))

        # data
        zz = griddata(
            points, model_data_density.zdraws, (grid_x, grid_y), method="linear"
        )
        cp = self.plot.axs[0, 2].contourf(grid_x, grid_y, zz, contourlines)
        CS = self.plot.axs[0, 2].contour(
            grid_x, grid_y, zz, contourlines, colors="black"
        )
        self.plot.axs[0, 2].clabel(CS, inline=1, colors="black")
        self.plot.axs[0, 2].set_title("Data: Fraction of positions leading to a draw")
        ModelFit.normalized_axis(self.plot.axs[0, 2], self.args.NormalizeToPawnValue)

        # model
        if self.args.modelFitting != "None":
            zwins = []
            for i in range(0, len(model_data_density.xs)):
                zwins.append(
                    fit.wdl(
                        model_data_density.xs[i],
                        model_data_density.ys[i],
                        model.popt_as,
                        model.popt_bs,
                    )[1]
                    / 1000.0
                )
            zz = griddata(points, zwins, (grid_x, grid_y), method="linear")
            cp = self.plot.axs[1, 2].contourf(grid_x, grid_y, zz, contourlines)
            CS = self.plot.axs[1, 2].contour(
                grid_x, grid_y, zz, contourlines, colors="black"
            )
            self.plot.axs[1, 2].clabel(CS, inline=1, colors="black")
            self.plot.axs[1, 2].set_title(
                "Model: Fraction of positions leading to a draw"
            )
            ModelFit.normalized_axis(
                self.plot.axs[1, 2], self.args.NormalizeToPawnValue
            )

        self.plot.fig.align_labels()

        self.plot.save(self.args.plot)


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
        # hide ugly parts for now TODO
        args.yPlotMin = (
            max(10, args.yDataMin) if args.yData == "move" else args.yDataMin
        )

    tic = time.time()

    data_loader = DataLoader(
        args.filename, args.NormalizeToPawnValue, args.NormalizeData
    )

    win, draw, loss, ntpv = data_loader.extract_wdl(
        data_loader.load_json(),
        args.moveMin,
        args.moveMax,
        args.NormalizeToPawnValue,
        args.NormalizeData,
        args.yData,
    )

    if args.modelFitting != "None":
        title = "Summary of win-draw-loss model analysis"
    else:
        title = "Summary of win-draw-loss data"

    # a hack to pass NormalizeToPawnValue to WdlModel TODO
    args.NormalizeToPawnValue = ntpv

    wdl_model = WdlModel(args, WdlPlot(title, args.pgnName))

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
        else ModelData([], [], [], [], [], "", "")
    )

    if args.plot != "no":
        wdl_model.create_plot(
            model_data_density,
            model,
        )

    if args.plot != "save+show":
        print(f"Total elapsed time = {time.time() - tic:.2f}s.")
