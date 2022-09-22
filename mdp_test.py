from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import numpy as np
import json
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
)
from itertools import product

# from timeit import timeit
import pickle
from os.path import exists
import mdptoolbox


def to_cartesian(r, theta):
    return r * np.cos(theta), r * np.sin(theta)


# distance/angle space -> x/y space?
def sample_shots(
    x_bins,
    y_bins,
    start_x,
    start_y,
    skill,
    distance,
    angle,
    in_sand,
    num_samples=100,
):
    distance_dev = distance / skill
    angle_dev = 1 / (2 * skill)

    if in_sand:
        distance_dev *= 2
        angle_dev *= 2

    # dist = multivariate_normal(
    #     mean=[distance, angle], cov=[distance_dev**2, angle_dev**2]
    # )
    # samples = dist.rvs(size=num_samples)
    # ds = [t[0] for t in samples]
    # angles = [t[1] for t in samples]
    dist_rv = norm(loc=distance, scale=distance_dev)
    angle_rv = norm(loc=angle, scale=angle_dev)
    ds = dist_rv.rvs(size=num_samples)
    angles = angle_rv.rvs(size=num_samples)
    xs, ys = to_cartesian(ds, angles)
    xs += start_x
    ys += start_y

    H, _, _ = np.histogram2d(
        xs,
        ys,
        # Transform edge bins to inf to capture everything falling off the map
        [
            np.concatenate([[-np.inf], x_bins[1:], [np.inf]]),
            np.concatenate([[-np.inf], y_bins[1:], [np.inf]]),
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

print(f"Map boundaries: x {map_min_x}, {map_max_x} y {map_min_y} {map_max_y}")

x_quant = 25
y_quant = 25
# Distance rating: 200 + s
dist_quant = 10
angle_quant = 16

x_tick = (map_max_x - map_min_x) / x_quant
y_tick = (map_max_y - map_min_y) / y_quant

min_x = map_min_x - x_tick
max_x = map_max_x + x_tick
min_y = map_min_y - y_tick
max_y = map_max_y + y_tick

print(f"Bin boundaries: x {min_x}, {max_x} y {min_y} {max_y}")

width = max_x - min_x
height = max_y - min_y

plt.gca().set_aspect(width / height)


x_bins = np.linspace(min_x, max_x, x_quant + 2, endpoint=False)
y_bins = np.linspace(min_y, max_y, y_quant + 2, endpoint=False)
# x_bins = np.arange(min_x, max_x, x_tick)
# y_bins = np.arange(min_y, max_y, y_tick)

# Visualize bins
# ==============
#
for x in x_bins:
    plt.axvline(x=x, color="black", alpha=0.1)
for y in y_bins:
    plt.axhline(y=y, color="black", alpha=0.1)
print(f"x range: [{x_bins[0]}, {x_bins[-1]}], y range:[{y_bins[0]}, {y_bins[-1]}]")

green_poly = ShapelyPolygon(map["map"])
sand_polys = [ShapelyPolygon(coords) for coords in map["sand traps"]]
plt.fill(*list(zip(*map["map"])), facecolor="#bbff66", edgecolor="black", linewidth=1)
start_x, start_y = map["start"]
plt.plot(start_x, start_y, "b.")
target_x, target_y = map["target"]
plt.plot(target_x, target_y, "r.")

if "sand traps" in map:
    for trap in map["sand traps"]:
        plt.fill(
            *list(zip(*trap)),
            facecolor="#ffffcc",
            edgecolor="black",
            linewidth=1,
        )

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

# visualize land
# ==============
#
# X, Y = np.meshgrid(x_bins, y_bins)
# plt.pcolormesh(
#     X,
#     Y,
#     is_land,
#     alpha=0.5,
# )

# debugging bin sizes
# print(len(x_bins), min_x, x_tick)
# plt.clf()
# plt.plot(x_bins, "b.")
# plt.plot(np.linspace(min_x, max_x, x_quant + 2) + 50, "r.")
# plt.savefig("map.png")
# for i in range(len(x_bins) - 1):
#     x = x_bins[i] + (0.5 * x_tick)
#     xi = int((x - min_x) / (x_tick))
#     print(
#         i,
#         "\t",
#         np.round(x_bins[i]),
#         "\t",
#         np.round(x),
#         "\t",
#         xi,
#         "\t",
#         x - (i * x_tick),
#         "\t",
#         x_bins[xi + 1] - x_bins[xi],
#     )


# print(len(y_bins))
# for yi in range(len(y_bins)):
#     y = y_bins[yi] + 0.5 * y_tick
#     print(y, int((y - min_y) / (y_tick)))


def to_bin(x, y):
    xi = int((x - min_x) / (x_tick))
    yi = int((y - min_y) / (y_tick))
    return xi, yi


def transition_histogram(start_x, start_y, skill, distance, angle, is_sand):
    H = sample_shots(x_bins, y_bins, start_x, start_y, skill, distance, angle, is_sand)

    transition = H.T
    start_xi, start_yi = to_bin(start_x, start_y)

    # print("start:", start_x, start_y, start_xi, start_yi)
    return_shots = 0
    for xi in range(x_quant + 2):
        for yi in range(y_quant + 2):
            if np.isnan(transition[yi][xi]):
                transition[yi][xi] = 0
            if not is_land[yi][xi]:
                # print(start_yi, start_xi, yi, xi)
                return_shots += transition[yi][xi]
                transition[yi][xi] = 0

    transition[start_yi][start_xi] += return_shots

    # normalize transition probabilities
    # if np.sum(transition) == 0:
    #     print(np.sum(H), start_x, start_y, distance, angle)
    transition = transition / max(np.sum(transition), 1)

    if np.sum(transition) == 0:
        print(np.sum(H), start_x, start_y, distance, angle)

    return transition


skill = 100
distance_levels = np.linspace(1, 200 + skill, dist_quant)
angle_levels = np.linspace(0, 2 * np.pi, angle_quant)

unreachable_transition = [[0 for _ in range(x_quant + 2)] for _ in range(y_quant + 2)]
unreachable_transition[0][0] = 1

# visualize shot
# ==============
#
# start_x, start_y = map["start"]
test_x = 26.4
test_y = 208.6
test_dist = 1.0
test_angle = 0.0

test_xi, test_yi = to_bin(test_x, test_y)

# H = sample_shots(
#     x_bins,
#     y_bins,
#     test_x,
#     test_y,
#     skill,
#     test_dist,
#     test_angle,
#     is_sand[test_yi][test_xi],
# )
# plt.plot(test_x, test_y, "g.")
# print(is_land[test_yi][test_xi])
# X, Y = np.meshgrid(x_bins, y_bins)
# print(np.sum(H))
# plt.pcolormesh(X, Y, H.T, alpha=0.8)

# start_x, start_y = map["start"]
# transition = transition_histogram(start_x, start_y, 10, 100, 3 * np.pi / 2, False)

# X, Y = np.meshgrid(x_bins, y_bins)
# plt.pcolormesh(X, Y, transition, alpha=0.8)

# plt.savefig("map.png")


S = list(product(x_bins, y_bins))
print("states:", len(S))
print("actions", len(distance_levels) * len(angle_levels))
if exists("T.pkl"):
    with open("T.pkl", "rb") as f:
        T = pickle.load(f)
else:
    T = [
        [
            # TODO: flatten into state transition probabilities
            transition_histogram(
                x + 0.5 * x_tick,
                y + 0.5 * y_tick,
                skill,
                distance,
                angle,
                is_sand[int((y - min_y) / y_tick)][int((x - min_x) / x_tick)],
            )
            if is_land[to_bin(x, y)[1]][to_bin(x, y)[0]]
            else unreachable_transition
            for x, y in S
        ]
        for distance in distance_levels
        for angle in angle_levels
    ]
    with open("T.pkl", "wb") as f:
        pickle.dump(T, f)
print(len(T))
print(len(T[0]))
# Profiling how long generating T takes
# def gen_action_transitions():
#     [
#         transition_histogram(
#             x + 0.5 * x_tick,
#             y + 0.5 * y_tick,
#             skill,
#             10,
#             0,
#             is_sand[int((y - min_y) / y_tick)][int((x - min_x) / x_tick)],
#         )
#         for x, y in S
#     ]


# print(timeit(gen_action_transitions, number=10))

T = np.array(
    [
        [[cell for row in histogram for cell in row] for histogram in action]
        for action in T
    ]
)

R = np.array([-1 for _ in range(x_quant + 2) for _ in range(y_quant + 2)])
target_x, target_y = to_bin(*map["target"])
R[target_y * (x_quant + 2) + target_x] = 10
print(R)

mdp = mdptoolbox.mdp.ValueIteration(T, R, 0.99)
mdp.run()
print("Converged in", mdp.time)
print(mdp.V)
print(mdp.policy)

v_hist = np.transpose(np.split(np.array(mdp.V), y_quant + 2))
X, Y = np.meshgrid(x_bins, y_bins)
plt.pcolormesh(X, Y, v_hist, alpha=0.8)

plt.savefig("map.png")


# Visualize values
