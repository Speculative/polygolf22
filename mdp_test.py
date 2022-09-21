from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import numpy as np
import json
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
)

skill = 10
# normally distributed with stddev d/s
distance = 40
# normally distributed with stddev 1/(2s)
angle = np.pi / 4


def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# distance/angle space -> x/y space?
distance_dev = distance / skill
angle_dev = 1 / (2 * skill)
dist = multivariate_normal(
    mean=[distance, angle], cov=[distance_dev**2, angle_dev**2]
)
samples = dist.rvs(size=10000)
ds1 = [t[0] for t in samples]
angles1 = [t[1] for t in samples]
xs1, ys1 = to_cartesian(ds1, angles1)

plt.subplot(121)
plt.scatter(xs1, ys1, 1)

H, xedges, yedges = np.histogram2d(
    xs1, ys1, [list(range(15, 45, 5)), list(range(15, 45, 5))]
)
plt.subplot(122)
plt.imshow(
    H,
    interpolation="nearest",
    origin="lower",
    extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
)

plt.savefig("render.png")

plt.clf()
plt.gca().invert_yaxis()
map_file = "maps/g5/zigzag_g5.json"
with open(map_file) as f:
    map = json.loads(f.read())

xs = [x for x, _ in map["map"]]
ys = [y for _, y in map["map"]]

min_x = min(xs)
max_x = max(xs)
min_y = min(ys)
max_y = max(ys)

width = max(xs) - min(xs)
height = max(ys) - min(ys)

plt.gca().set_aspect(width / height)

x_quant = 50
y_quant = 50
dist_quant = 10
angle_quant = 8

x_bins = np.linspace(min_x, max_x, x_quant)
for x in x_bins:
    plt.axvline(x=x, color="black", alpha=0.1)
y_bins = np.linspace(min_y, max_y, y_quant)
for y in y_bins:
    plt.axhline(y=y, color="black", alpha=0.1)
print(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])

green_poly = ShapelyPolygon(map["map"])
plt.fill(*list(zip(*map["map"])), facecolor="#bbff66", edgecolor="black", linewidth=1)
start_x, start_y = map["start"]
plt.plot(start_x, start_y, "bo")
target_x, target_y = map["target"]
plt.plot(target_x, target_y, "ro")

if "sand traps" in map:
    for trap in map["sand traps"]:
        plt.fill(
            *list(zip(*trap)),
            facecolor="#ffffcc",
            edgecolor="black",
            linewidth=1,
        )

plt.savefig("map.png")
