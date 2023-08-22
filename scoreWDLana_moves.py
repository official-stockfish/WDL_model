from collections import Counter
import json
from ast import literal_eval
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import griddata


with open("scoreWDLstat.json", "r") as infile:
    inputdata = json.load(infile)

print("read data")

inpdict = {literal_eval(k): v for k, v in inputdata.items()}

win = Counter()
loss = Counter()
draw = Counter()
for k, v in inpdict.items():
    (result, move, material, score) = k
    if score < -400 or score > 400:
        continue
    if move < 10 or move > 120:
        continue
    if result == "W":
        win[(score, move)] += v
    elif result == "L":
        loss[(score, move)] += v
    elif result == "D":
        draw[(score, move)] += v

print("counted")

coords = list(
    set(k for k in win).union(set(k for k in loss)).union(set(k for k in draw))
)
coords.sort()
xs = []
ys = []
zs = []
for coord in coords:
    total = float(win[coord] + loss[coord] + draw[coord])
    x, y = coord
    xs.append(x)
    ys.append(y)
    zs.append(draw[coord] / total)

print("processing done, plotting")
font = {"family": "DejaVu Sans", "weight": "normal", "size": 20}
grid_x, grid_y = np.mgrid[-400:400:40j, 10:120:22j]
points = np.array(list(zip(xs, ys)))
zz = griddata(points, zs, (grid_x, grid_y), method="cubic")
fig = plt.figure(figsize=(6, 5))
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
    "Fraction of positions, with a given move number and score, leading to a draw"
)
ax.set_xlabel("score")
ax.set_ylabel("move")
ax.yaxis.grid(True)
ax.xaxis.grid(True)
plt.show()
