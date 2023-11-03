import json, argparse, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval
from scipy.interpolate import griddata
from scipy.optimize import curve_fit, minimize
from dataclasses import dataclass
from typing import Literal, Callable, Any
import math
from time import time


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

    xs: list[float]  # scores
    ys: list[int]  # moves or material
    zwins: list[float]  # corresponding win probability
    zdraws: list[float]  # draw prob
    zlosses: list[float]  # loss prob


class DataLoader:
    def __init__(self, filenames: list[str]):
        self.filenames = filenames

    def load_json(self) -> dict[str, int]:
        """Load the json, in which the key describes the position (result, move, material, eval),
        and in which the value is the observed count of these positions
        """
        inputdata: dict[str, int] = {}
        for filename in self.filenames:
            print(f"Reading score stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)

                for key, value in data.items():
                    inputdata[key] = inputdata.get(key, 0) + value
        return inputdata

    def extract_wdl(
        self,
        inputdata: dict[str, int],
        moveMin: int,
        moveMax: int,
        NormalizeToPawnValue: int,
        yDataFormat: Literal["move", "material"],
    ) -> tuple[
        Counter[tuple[float, int]],
        Counter[tuple[float, int]],
        Counter[tuple[float, int]],
    ]:
        """Extract three arrays, win draw and loss, counting positions with a given score (float) and move/material (int) that are wdl"""
        inpdict: dict[tuple[str, int, int, int], int] = {
            literal_eval(k): v for k, v in inputdata.items()
        }

        win: Counter[tuple[float, int]] = Counter()
        draw: Counter[tuple[float, int]] = Counter()
        loss: Counter[tuple[float, int]] = Counter()
        # filter out (score, yData) WDL data (i.e. material or move summed out)
        for (result, move, material, score), v in inpdict.items():
            # exclude large scores and unwanted move numbers
            if abs(score) > 400 or move < moveMin or move > moveMax:
                continue

            # convert the cp score to the internal value
            score_internal = score * NormalizeToPawnValue / 100

            yData = move if yDataFormat == "move" else material

            if result == "W":
                win[score_internal, yData] += v
            elif result == "D":
                draw[score_internal, yData] += v
            elif result == "L":
                loss[score_internal, yData] += v

        print(
            f"Retained (W,D,L) = ({sum(win.values())}, {sum(draw.values())}, {sum(loss.values())}) positions."
        )
        return win, draw, loss

    def get_model_data_density(
        self,
        win: Counter[tuple[float, int]],
        draw: Counter[tuple[float, int]],
        loss: Counter[tuple[float, int]],
    ) -> ModelDataDensity:
        """Turns the counts of positions in densities/frequencies

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
# Model to be fitted in order to predict the winrate from score and move (or material).
#
# This defines the model functions
#
class ModelFit:
    def __init__(self, y_data_target: int, normalize_to_pawn_value: int):
        self.y_data_target = y_data_target
        self.normalize_to_pawn_value = normalize_to_pawn_value

    @staticmethod
    def winmodel(x, a, b):
        def skip_overflow(arg):
            if arg > 0:
                return np.exp(-arg) / (1.0 + np.exp(-arg))
            else:
                return 1.0 / (1.0 + np.exp(arg))

        if type(x) == np.ndarray:
            res = []
            for xs in x:
                arg = -(xs - a) / b
                res.append(skip_overflow(arg))
            return np.array(res)
        else:
            arg = -(x - a) / b
            return skip_overflow(arg)

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

    def poly3(self, x: float | list[float], a, b, c, d) -> float:
        """compute the value of a polynomial of 3rd order in a point x"""
        xnp = np.asarray(x) / self.y_data_target
        return ((a * xnp + b) * xnp + c) * xnp + d

    def poly3_str(self, coeffs) -> str:
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(
                val for pair in zip(coeffs, [self.y_data_target] * 4) for val in pair
            )[:-1]
        )

    def wdl(
        self,
        score: float,
        move_or_material: int,
        popt_as: list[float],
        popt_bs: list[float],
    ) -> tuple[int, int, int]:
        """Compute the integer wdl (per-mille) using polynomial approximation for a and b"""
        a = self.poly3(move_or_material, *popt_as)
        b = self.poly3(move_or_material, *popt_bs)
        w = int(1000 * ModelFit.winmodel(score, a, b))
        l = int(1000 * ModelFit.winmodel(-score, a, b))
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
        if len(asbs) == 8:
            popt_as = asbs[0:4]
            popt_bs = asbs[4:8]
            a = self.fit.poly3(mom, *popt_as)
            b = self.fit.poly3(mom, *popt_bs)
        else:
            a = asbs[0]
            b = asbs[1]

        return a, b

    def estimateScore(self, asbs: list[float], score, mom):
        """Estimate game score based on probability of WDL"""

        a, b = self.get_ab(asbs, mom)

        # guard against unphysical stuff
        if a <= 0 or b <= 0:
            return 4

        probw = ModelFit.winmodel(score, a, b)
        probl = ModelFit.winmodel(-score, a, b)
        probd = 1 - probw - probl
        return probw + 0.5 * probd + 0

    def scoreError(self, asbs: list[float]):
        """Sum of the squared error on the game score"""
        scoreErr = 0

        for score, mom in self.win.keys():
            scoreErr += (
                self.win[score, mom] * (self.estimateScore(asbs, score, mom) - 1) ** 2
            )

        for score, mom in self.loss.keys():
            scoreErr += (
                self.loss[score, mom] * (self.estimateScore(asbs, score, mom) - 0) ** 2
            )

        for score, mom in self.draw.keys():
            scoreErr += (
                self.draw[score, mom]
                * (self.estimateScore(asbs, score, mom) - 0.5) ** 2
            )

        return scoreErr

    def evalLogProbability(self, asbs: list[float]):
        """-log(product of game outcome probability)"""
        evalLogProb = 0

        for score, mom in self.win.keys():
            count = self.win[score, mom]
            a, b = self.get_ab(asbs, mom)
            prob = ModelFit.winmodel(score, a, b)
            if prob < 1e-14:
                prob = 1e-14
            evalLogProb += count * math.log(prob)

        for score, mom in self.loss.keys():
            count = self.loss[score, mom]
            a, b = self.get_ab(asbs, mom)
            prob = ModelFit.winmodel(-score, a, b)
            if prob < 1e-14:
                prob = 1e-14
            evalLogProb += count * math.log(prob)

        for score, mom in self.draw.keys():
            count = self.draw[score, mom]
            a, b = self.get_ab(asbs, mom)
            probw = ModelFit.winmodel(score, a, b)
            probl = ModelFit.winmodel(-score, a, b)
            prob = 1 - (probw + probl)
            if prob < 1e-14:
                prob = 1e-14
            evalLogProb += count * math.log(prob)

        return -evalLogProb


@dataclass
class ModelData:
    popt_as: list[float]
    popt_bs: list[float]
    model_ms: list[float]
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
        xdata: list[float],
        ywindata: list[float],
        ydrawdata: list[float],
        ylossdata: list[float],
        popt,
    ):
        # plot sample curve at yDataTarget
        self.plot.axs[0, 0].plot(xdata, ywindata, "b.", label="Measured winrate")
        self.plot.axs[0, 0].plot(xdata, ydrawdata, "g.", label="Measured drawrate")
        self.plot.axs[0, 0].plot(xdata, ylossdata, "c.", label="Measured lossrate")

        ymodel = []
        for x in xdata:
            ymodel.append(ModelFit.winmodel(x, popt[0], popt[1]))
        self.plot.axs[0, 0].plot(xdata, ymodel, "r-", label="Model")

        ymodel = []
        for x in xdata:
            ymodel.append(ModelFit.winmodel(-x, popt[0], popt[1]))
        self.plot.axs[0, 0].plot(xdata, ymodel, "r-")

        ymodel = []
        for x in xdata:
            ymodel.append(
                1
                - ModelFit.winmodel(x, popt[0], popt[1])
                - ModelFit.winmodel(-x, popt[0], popt[1])
            )
        self.plot.axs[0, 0].plot(xdata, ymodel, "r-")

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
        func: Callable[[list[float], list[float], list[float], list[float], Any], None],
    ):
        scores, moms, winrate, drawrate, lossrate = xs, ys, zwins, zdraws, zlosses

        model_ms, model_as, model_bs = [], [], []

        grouping = 1

        # mom = move or material, depending on self.args.yData
        for mom in range(self.args.yDataMin, self.args.yDataMax + 1, grouping):
            mmin, mmax = mom, mom + grouping
            xdata, ywindata, ydrawdata, ylossdata = [], [], [], []
            for i in range(0, len(moms)):
                if moms[i] < mmin or moms[i] >= mmax:
                    continue
                xdata.append(scores[i])
                ywindata.append(winrate[i])
                ydrawdata.append(drawrate[i])
                ylossdata.append(lossrate[i])

            # skip fit for move counts with very few data points
            if len(ywindata) < 10:
                continue

            popt: tuple[float, float]

            # Get initial values based on a simple fit of the curve.
            popt, pcov = curve_fit(
                ModelFit.winmodel,
                xdata,
                ywindata,
                p0=[self.args.NormalizeToPawnValue, self.args.NormalizeToPawnValue / 6],
            )

            # Refine result based on data, optimizing an objective function

            # get the subset of data relevant for this mom
            if self.args.modelFitting != "fitDensity":
                winsubset: Counter[tuple[float, int]] = Counter()
                drawsubset: Counter[tuple[float, int]] = Counter()
                losssubset: Counter[tuple[float, int]] = Counter()
                for score, momkey in win.keys():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    winsubset[score, momkey] = win[score, momkey]
                for score, momkey in loss.keys():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    losssubset[score, momkey] = loss[score, momkey]
                for score, momkey in draw.keys():
                    if momkey < mmin or momkey >= mmax:
                        continue
                    drawsubset[score, momkey] = draw[score, momkey]

                # Miniminize the objective function
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
                    popt,
                    method="Powell",
                    options={"maxiter": 100000, "disp": False, "xtol": 1e-6},
                )
                popt = res.x

            # Prepare output

            # store result.
            model_ms.append(mom)
            model_as.append(popt[0])
            model_bs.append(popt[1])

            # TODO ... this shows the 'local fit, it probably would be interesting to show the final result,
            # based on the a,b obtained from the fit.
            if self.args.plot != "no" and mom == self.args.yDataTarget and func != None:
                func(xdata, ywindata, ydrawdata, ylossdata, popt)

        return model_as, model_bs, model_ms

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
        # convert to model, fit the winmodel a and b,
        # for a given value of the move/material counter
        #

        fit = ModelFit(self.args.yDataTarget, self.args.NormalizeToPawnValue)

        model_as, model_bs, model_ms = self.extract_model_data(
            xs, ys, zwins, zdraws, zlosses, win, draw, loss, fit, self.sample_curve_y
        )

        #
        # now capture the functional behavior of a and b as a function of the move counter
        # simple polynomial fit
        #

        # fit a and b
        popt_as, pcov = curve_fit(fit.poly3, model_ms, model_as)
        popt_bs, pcov = curve_fit(fit.poly3, model_ms, model_bs)

        # Refinement phase
        #
        # optimize the likelihood of seeing the data ...
        #    our model_as, model_bs / popt_as, popt_bs are just initial guesses.
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
            popt_as = res.x[0:4]
            popt_bs = res.x[4:8]
            popt_all = res.x
            print("Final objective function:   ", objectiveFunction(popt_all))
            print(res.message)

        # Prepare output
        label_as, label_bs = "as = " + fit.poly3_str(popt_as), "bs = " + fit.poly3_str(
            popt_bs
        )

        #
        # now we can define the conversion factor from internal score to centipawn such that
        # an expected win score of 50% is for a score of 'a', we pick this value for the yDataTarget
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

        # give as output as well
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
                fit.poly3(model.model_ms, *model.popt_as),
                "r-",
                label="fit: " + model.label_as,
            )
            self.plot.axs[1, 0].plot(model.model_ms, model.model_bs, "g.", label="bs")
            self.plot.axs[1, 0].plot(
                model.model_ms,
                fit.poly3(model.model_ms, *model.popt_bs),
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
        default=328,
        help="Value needed for converting the games' cp scores to the SF's internal score.",
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
        "--modelFitting",
        choices=["fitDensity", "optimizeProbability", "optimizeScore", "None"],
        default="fitDensity",
        help="Choice of model fitting: Fit the win rate curves, maximimize the probability of predicting the outcome, minimize the squared error in predicted score, or no fitting.",
    )

    args = parser.parse_args()

    if args.yData == "material":
        # fix default values for material
        if args.yDataMax == 120 and args.yDataMin == 3:
            args.yDataMin, args.yDataMax = 10, 78

    if args.yPlotMin is None:
        # hide ugly parts for now TODO
        args.yPlotMin = (
            max(10, args.yDataMin) if args.yData == "move" else args.yDataMin
        )

    data_loader = DataLoader(args.filename)

    print(f"Converting scores with NormalizeToPawnValue = {args.NormalizeToPawnValue}.")
    tic = time()

    win, draw, loss = data_loader.extract_wdl(
        data_loader.load_json(),
        args.moveMin,
        args.moveMax,
        args.NormalizeToPawnValue,
        args.yData,
    )

    if args.modelFitting != "None":
        title = "Summary of win-draw-loss model analysis"
        pgnName = "WDL_model_summary.png"
    else:
        title = "Summary of win-draw-loss data"
        pgnName = f"WDL_data_{args.yData}.png"

    wdl_model = WdlModel(args, WdlPlot(title, pgnName))

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
        print(f"Total elapsed time = {time() - tic:.2f}s.")
