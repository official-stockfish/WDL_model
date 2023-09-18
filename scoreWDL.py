import json, argparse, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval
from scipy.interpolate import griddata
from scipy.optimize import curve_fit


class WdlPlot:
    def __init__(self, title, pgnName):
        self.title = title
        self.pgnName = pgnName

        self.fig, self.axs = plt.subplots(
            2, 3, figsize=(11.69 * 1.5, 8.27 * 1.5), constrained_layout=True
        )
        self.fig.suptitle(self.title, fontsize="x-large")

    def save(self, plot_setting):
        plt.savefig(self.pgnName, dpi=300)
        if plot_setting == "save+show":
            plt.show()
        plt.close()
        print(f"Saved graphics to {self.pgnName}.")


class DataLoader:
    def __init__(self, filenames):
        self.filenames = filenames

    def load_json(self):
        inputdata = {}
        for filename in self.filenames:
            print(f"Reading score stats from {filename}.")
            with open(filename) as infile:
                data = json.load(infile)

                for key, value in data.items():
                    inputdata[key] = inputdata.get(key, 0) + value
        return inputdata

    def extract_wdl(
        self, inputdata, moveMin, moveMax, NormalizeToPawnValue, yDataFormat
    ):
        inpdict = {literal_eval(k): v for k, v in inputdata.items()}
        win, draw, loss = Counter(), Counter(), Counter()
        # filter out (score, yData) WDL data (i.e. material or move summed out)
        for (result, move, material, score), v in inpdict.items():
            # exclude large scores and unwanted move numbers
            if abs(score) > 400 or move < moveMin or move > moveMax:
                continue

            # convert the cp score to the internal value
            score = score * NormalizeToPawnValue / 100

            yData = move if yDataFormat == "move" else material

            if result == "W":
                win[score, yData] += v
            elif result == "D":
                draw[score, yData] += v
            elif result == "L":
                loss[score, yData] += v

        print(
            f"Retained (W,D,L) = ({sum(win.values())}, {sum(draw.values())}, {sum(loss.values())}) positions."
        )
        return win, draw, loss

    def convert_to_model(self, win, draw, loss):
        coords = sorted(set(list(win.keys()) + list(draw.keys()) + list(loss.keys())))
        xs, ys, zwins, zdraws, zlosses = [], [], [], [], []
        for x, y in coords:
            xs.append(x)
            ys.append(y)
            total = win[x, y] + draw[x, y] + loss[x, y]
            zwins.append(win[x, y] / total)
            zdraws.append(draw[x, y] / total)
            zlosses.append(loss[x, y] / total)
        return xs, ys, zwins, zdraws, zlosses


#
# fit a model to predict winrate from score and move
# define model functions
#


class ModelFit:
    def __init__(self, y_data_target, normalize_to_pawn_value):
        self.y_data_target = y_data_target
        self.normalize_to_pawn_value = normalize_to_pawn_value

    def winmodel(x, a, b):
        return 1.0 / (1.0 + np.exp(-(x - a) / b))

    def poly3(self, x, a, b, c, d):
        xnp = np.asarray(x) / self.y_data_target
        return ((a * xnp + b) * xnp + c) * xnp + d

    def poly3_str(self, coeffs):
        return (
            "((%5.3f * x / %d + %5.3f) * x / %d + %5.3f) * x / %d + %5.3f"
            % tuple(
                val for pair in zip(coeffs, [self.y_data_target] * 4) for val in pair
            )[:-1]
        )

    def wdl(self, score, move_or_material, popt_as, popt_bs):
        a = self.poly3(move_or_material, *popt_as)
        b = self.poly3(move_or_material, *popt_bs)
        w = int(1000 * ModelFit.winmodel(score, a, b))
        l = int(1000 * ModelFit.winmodel(-score, a, b))
        d = 1000 - w - l
        return w, d, l

    def normalized_axis(self, ax):
        ax2 = ax.twiny()
        tickmin = int(np.ceil(ax.get_xlim()[0] / self.normalize_to_pawn_value)) * 2
        tickmax = int(np.floor(ax.get_xlim()[1] / self.normalize_to_pawn_value)) * 2 + 1
        new_tick_locations = np.array(
            [x / 2 * self.normalize_to_pawn_value for x in range(tickmin, tickmax)]
        )

        def tick_function(X):
            V = X / self.normalize_to_pawn_value
            return [(f"{z:.0f}" if z % 1 < 0.1 else "") for z in V]

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))


class WdlModel:
    def __init__(self, args, plot):
        self.args = args
        self.plot = plot

    def fit_model(self, xs, ys, zwins, zdraws, zlosses):
        print(f"Fit WDL model based on {self.args.yData}.")
        #
        # convert to model, fit the winmodel a and b,
        # for a given value of the move/material counter
        #
        scores, moms, winrate, drawrate, lossrate = xs, ys, zwins, zdraws, zlosses

        model_ms, model_as, model_bs = [], [], []

        grouping = 1

        fit = ModelFit(args.yDataTarget, self.args.NormalizeToPawnValue)

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

            popt, pcov = curve_fit(
                ModelFit.winmodel,
                xdata,
                ywindata,
                p0=[self.args.NormalizeToPawnValue, self.args.NormalizeToPawnValue / 6],
            )
            model_ms.append(mom)
            model_as.append(popt[0])
            model_bs.append(popt[1])

            # plot sample curve at yDataTarget
            if self.args.plot != "no" and mom == self.args.yDataTarget:
                self.plot.axs[0, 0].plot(
                    xdata, ywindata, "b.", label="Measured winrate"
                )
                self.plot.axs[0, 0].plot(
                    xdata, ydrawdata, "g.", label="Measured drawrate"
                )
                self.plot.axs[0, 0].plot(
                    xdata, ylossdata, "c.", label="Measured lossrate"
                )

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

                fit.normalized_axis(self.plot.axs[0, 0])

        #
        # now capture the functional behavior of a and b as a function of the move counter
        # simple polynomial fit
        #

        # fit a and b
        popt_as, pcov = curve_fit(fit.poly3, model_ms, model_as)
        popt_bs, pcov = curve_fit(fit.poly3, model_ms, model_bs)
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

        if self.args.plot != "no":
            # graphs of a and b as a function of move/material
            print("Plotting move/material dependence of model parameters.")
            self.plot.axs[1, 0].plot(model_ms, model_as, "b.", label="as")
            self.plot.axs[1, 0].plot(
                model_ms,
                fit.poly3(model_ms, *popt_as),
                "r-",
                label="fit: " + label_as,
            )
            self.plot.axs[1, 0].plot(model_ms, model_bs, "g.", label="bs")
            self.plot.axs[1, 0].plot(
                model_ms,
                fit.poly3(model_ms, *popt_bs),
                "m-",
                label="fit: " + label_bs,
            )

            self.plot.axs[1, 0].set_xlabel(self.args.yData)
            self.plot.axs[1, 0].set_ylabel("parameters (in internal value units)")
            self.plot.axs[1, 0].legend(fontsize="x-small")
            self.plot.axs[1, 0].set_title("Winrate model parameters")
            self.plot.axs[1, 0].set_ylim(bottom=0.0)

        return popt_as, popt_bs

    def create_plot(
        self,
        popt_as,
        popt_bs,
        xs,
        ys,
        zwins,
        zdraws,
    ):
        # now generate contour plots
        contourlines = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97, 1.0]

        print("Processing done, plotting 2D data.")
        ylabelStr = self.args.yData + " (1,3,3,5,9)" * bool(
            self.args.yData == "material"
        )
        for i in [0, 1]:
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
        ymin, ymax = self.args.yDataMin, self.args.yDataMax
        if self.args.yData == "move":
            ymin = max(10, self.args.yDataMin)  #  hide ugly parts for now TODO
        grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]
        points = np.array(list(zip(xs, ys)))

        # data
        zz = griddata(points, zwins, (grid_x, grid_y), method="linear")
        cp = self.plot.axs[0, 1].contourf(grid_x, grid_y, zz, contourlines)
        self.plot.fig.colorbar(cp, ax=self.plot.axs[:, -1], shrink=0.618)
        CS = self.plot.axs[0, 1].contour(
            grid_x, grid_y, zz, contourlines, colors="black"
        )
        self.plot.axs[0, 1].clabel(CS, inline=1, colors="black")
        self.plot.axs[0, 1].set_title("Data: Fraction of positions leading to a win")

        fit = ModelFit(args.yDataTarget, self.args.NormalizeToPawnValue)

        fit.normalized_axis(self.plot.axs[0, 1])

        # model
        if self.args.fit:
            for i in range(0, len(xs)):
                zwins[i] = fit.wdl(xs[i], ys[i], popt_as, popt_bs)[0] / 1000.0
            zz = griddata(points, zwins, (grid_x, grid_y), method="linear")
            cp = self.plot.axs[1, 1].contourf(grid_x, grid_y, zz, contourlines)
            CS = self.plot.axs[1, 1].contour(
                grid_x, grid_y, zz, contourlines, colors="black"
            )
            self.plot.axs[1, 1].clabel(CS, inline=1, colors="black")
            self.plot.axs[1, 1].set_title(
                "Model: Fraction of positions leading to a win"
            )
            fit.normalized_axis(self.plot.axs[1, 1])

        # for draws, plot between -2 and 2 pawns, using a 30x22 grid
        xmin = -((2 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        xmax = ((2 * self.args.NormalizeToPawnValue) // 100 + 1) * 100
        grid_x, grid_y = np.mgrid[xmin:xmax:30j, ymin:ymax:22j]
        points = np.array(list(zip(xs, ys)))

        # data
        zz = griddata(points, zdraws, (grid_x, grid_y), method="linear")
        cp = self.plot.axs[0, 2].contourf(grid_x, grid_y, zz, contourlines)
        CS = self.plot.axs[0, 2].contour(
            grid_x, grid_y, zz, contourlines, colors="black"
        )
        self.plot.axs[0, 2].clabel(CS, inline=1, colors="black")
        self.plot.axs[0, 2].set_title("Data: Fraction of positions leading to a draw")
        fit.normalized_axis(self.plot.axs[0, 2])

        # model
        if self.args.fit:
            for i in range(0, len(xs)):
                zwins[i] = fit.wdl(xs[i], ys[i], popt_as, popt_bs)[1] / 1000.0
            zz = griddata(points, zwins, (grid_x, grid_y), method="linear")
            cp = self.plot.axs[1, 2].contourf(grid_x, grid_y, zz, contourlines)
            CS = self.plot.axs[1, 2].contour(
                grid_x, grid_y, zz, contourlines, colors="black"
            )
            self.plot.axs[1, 2].clabel(CS, inline=1, colors="black")
            self.plot.axs[1, 2].set_title(
                "Model: Fraction of positions leading to a draw"
            )
            fit.normalized_axis(self.plot.axs[1, 2])

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
        "--fit",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fit WDL model or not. Data contour plots are always created.",
    )
    parser.add_argument(
        "--plot",
        choices=["save+show", "save", "no"],
        default="save+show",
        help="Save/show graphics or not. Useful for batch processing.",
    )

    args = parser.parse_args()

    plot = None
    if args.yData == "material":
        # fix default values for material
        if args.yDataMax == 120 and args.yDataMin == 3:
            args.yDataMin, args.yDataMax = 10, 78

    if args.plot != "no":
        if args.fit:
            title = "Summary of win-draw-loss model analysis"
            pgnName = "WDL_model_summary.png"
        else:
            title = "Summary of win-draw-loss data"
            pgnName = f"WDL_data_{args.yData}.png"

        plot = WdlPlot(title, pgnName)

    data_loader = DataLoader(args.filename)

    print(f"Converting scores with NormalizeToPawnValue = {args.NormalizeToPawnValue}.")

    win, draw, loss = data_loader.extract_wdl(
        data_loader.load_json(),
        args.moveMin,
        args.moveMax,
        args.NormalizeToPawnValue,
        args.yData,
    )

    xs, ys, zwins, zdraws, zlosses = data_loader.convert_to_model(win, draw, loss)

    wdl_model = WdlModel(args, plot)

    popt_as, popt_bs = (
        wdl_model.fit_model(xs, ys, zwins, zdraws, zlosses)
        if args.fit
        else (None, None)
    )

    if args.plot != "no":
        wdl_model.create_plot(popt_as, popt_bs, xs, ys, zwins, zdraws)
