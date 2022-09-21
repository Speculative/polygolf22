from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import numpy as np
import json
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
)


def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# distance/angle space -> x/y space?
def sample_shots(
    x_bins, y_bins, start_x, start_y, skill, distance, angle, num_samples=10_000
):
    distance_dev = distance / skill
    angle_dev = 1 / (2 * skill)
    dist = multivariate_normal(
        mean=[distance, angle], cov=[distance_dev**2, angle_dev**2]
    )
    samples = dist.rvs(size=num_samples)
    ds = [t[0] for t in samples]
    angles = [t[1] for t in samples]
    xs, ys = to_cartesian(ds, angles)
    xs += start_x
    ys += start_y

    H, _, _ = np.histogram2d(
        xs,
        ys,
        # Transform edge bins to inf to capture everything falling off the map
        [
            np.concatenate([[-np.inf], x_bins[1:-1], [np.inf]]),
            np.concatenate([[-np.inf], y_bins[1:-1], [np.inf]]),
        ],
    )
    return H


plt.savefig("render.png")

plt.clf()
plt.gca().invert_yaxis()
map_file = "maps/g5/zigzag_g5.json"
with open(map_file) as f:
    map = json.loads(f.read())

map_xs = [x for x, _ in map["map"]]
map_ys = [y for _, y in map["map"]]

map_min_x = min(map_xs)
map_max_x = max(map_xs)
map_min_y = min(map_ys)
map_max_y = max(map_ys)

x_quant = 50
y_quant = 50
# Distance rating: 200 + s
dist_quant = 10
angle_quant = 8

x_tick = (map_max_x - map_min_x) / x_quant
y_tick = (map_max_y - map_min_y) / y_quant

min_x = map_min_x - x_tick
max_x = map_max_x + x_tick
min_y = map_min_y - y_tick
max_y = map_max_y + y_tick

width = max_x - min_x
height = max_y - min_y

plt.gca().set_aspect(width / height)


x_bins = np.linspace(map_min_x - x_tick, map_max_x + x_tick, x_quant + 2)
y_bins = np.linspace(map_min_y - y_tick, map_max_y + y_tick, y_quant + 2)

# Visualize bins
# ==============
#
for x in x_bins:
    plt.axvline(x=x, color="black", alpha=0.1)
for y in y_bins:
    plt.axhline(y=y, color="black", alpha=0.1)
print(x_bins[0], x_bins[-1], y_bins[0], y_bins[-1])

green_poly = ShapelyPolygon(map["map"])
sand_polys = [ShapelyPolygon(coords) for coords in map["sand traps"]]
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

# visualize shot
# ==============
#
# start_x, start_y = map["start"]
# H = sample_shots(x_bins, y_bins, start_x, start_y, 10, 210, 2 * np.pi)
# X, Y = np.meshgrid(x_bins, y_bins)
# plt.pcolormesh(X, Y, H.T, alpha=0.8)

# visualize land
# ==============
#
cell_polys = [
    [
        ShapelyPolygon(
            [
                (min_x + xi * x_tick, min_y + yi * y_tick),
                (min_x + (xi + 1) * x_tick, min_y + yi * y_tick),
                (min_x + (xi + 1) * x_tick, min_y + (yi + 1) * y_tick),
                (min_x + xi * x_tick, min_y + (yi + 1) * y_tick),
                (min_x + xi * x_tick, min_y + yi * y_tick),
            ]
        )
        for xi in range(x_quant + 2)
    ]
    for yi in range(y_quant + 2)
]

is_land = [[green_poly.contains(cell_poly) for cell_poly in row] for row in cell_polys]
is_sand = [
    [
        any(cell_poly.intersects(sand_poly) for sand_poly in sand_polys)
        for cell_poly in row
    ]
    for row in cell_polys
]

X, Y = np.meshgrid(x_bins, y_bins)
plt.pcolormesh(
    X,
    Y,
    is_sand,
    alpha=0.5,
)


plt.savefig("map.png")
