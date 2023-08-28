import json, argparse, numpy as np, matplotlib.pyplot as plt
from collections import Counter
from ast import literal_eval
from scipy.interpolate import griddata

parser = argparse.ArgumentParser(
    description="Create a win-rate contour plot in the (score, move) plane from fishtest game statistics.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
parser.add_argument(
    "filename",
    nargs="?",
    help="json file with fishtest games' WDL statistics",
    default="scoreWDLstat.json",
)
parser.add_argument(
    "--show",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show graphics or not. An image is always saved. Useful for batch processing.",
)
args = parser.parse_args()

print(f"Reading data from {args.filename}.")
with open(args.filename) as infile:
    inputdata = json.load(infile)

print("Filtering data.")
inpdict = {literal_eval(k): v for k, v in inputdata.items()}
win, draw, loss = Counter(), Counter(), Counter()
# filter out (score, material) WDL data (i.e. move summed out)
for (result, move, material, score), v in inpdict.items():
    # exclude large scores and the endings of very long games
    if abs(score) > 400 or move < 0 or move > 120:
        continue
    if result == "W":
        win[score, material] += v
    elif result == "D":
        draw[score, material] += v
    elif result == "L":
        loss[score, material] += v

print(
    f"Retained (W,D,L) = ({sum(win.values())}, {sum(draw.values())}, {sum(loss.values())}) positions."
)

# create (score, material) -> WDL ratio data
coords = sorted(set(list(win.keys()) + list(draw.keys()) + list(loss.keys())))
xs, ys, zwins = [], [], []
for x, y in coords:
    xs.append(x)
    ys.append(y)
    total = win[x, y] + loss[x, y] + draw[x, y]
    zwins.append(win[x, y] / total)

print("Processing done, plotting.")
font = {"family": "DejaVu Sans", "weight": "normal", "size": 20}
grid_x, grid_y = np.mgrid[-200:400:30j, 0:78:38j]
points = np.array(list(zip(xs, ys)))
zz = griddata(points, zwins, (grid_x, grid_y), method="cubic")
fig = plt.figure(figsize=(11.69 * 1.5, 8.27 * 1.5))
plt.rc("font", **font)
left, bottom, width, height = 0.1, 0.1, 0.8, 0.8
ax = fig.add_axes([left, bottom, width, height])
cp = plt.contourf(grid_x, grid_y, zz, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0])
plt.colorbar(cp)
CS = plt.contour(
    grid_x, grid_y, zz, [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0], colors="black"
)
ax.clabel(CS, inline=1, colors="black")
ax.set_title(
    "Fraction of positions, with a given material value and score, leading to a win"
)
ax.set_xlabel("score")
ax.set_ylabel("material (1,3,3,5,9)")
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.savefig("WDL_model_material.png", dpi=300)
if args.show:
    plt.show()
plt.close()
print("Saved graphics to WDL_model_material.png.")
