from scipy.stats import multivariate_normal, norm
import matplotlib.pyplot as plt
import numpy as np
import json
from shapely.geometry import (
    Polygon as ShapelyPolygon,
    Point as ShapelyPoint,
)
from itertools import product
from time import perf_counter
from scipy.integrate import dblquad
from timeit import timeit
import pickle
from os.path import exists
from os import makedirs
import mdptoolbox
import multiprocessing


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


plt.clf()
plt.gca().invert_yaxis()
map_file = "maps/default/simple_with_sandtraps.json"
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
dist_quant = 20
angle_quant = 36

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
total_x_bins = len(x_bins)
total_y_bins = len(y_bins)

# Profiling how long generating T takes
# =====================================
#
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


# print(
#     "Time to generate all histograms for 1 action:",
#     timeit(gen_action_transitions, number=1),
# )


# Visualize bins
# ==============
#
def draw_bins():
    for x in x_bins:
        plt.axvline(x=x, color="black", alpha=0.1)
    for y in y_bins:
        plt.axhline(y=y, color="black", alpha=0.1)


print(f"x range: [{x_bins[0]}, {x_bins[-1]}], y range:[{y_bins[0]}, {y_bins[-1]}]")

green_poly = ShapelyPolygon(map["map"])
sand_polys = [ShapelyPolygon(coords) for coords in map["sand traps"]]


def draw_map():
    plt.fill(
        *list(zip(*map["map"])), facecolor="#bbff66", edgecolor="black", linewidth=1
    )
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
        for xi in range(total_x_bins)
    ]
    for yi in range(total_y_bins)
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


def to_bin(x, y):
    xi = int((x - min_x) / (x_tick))
    yi = int((y - min_y) / (y_tick))
    return xi, yi

def joint_density(location, start, angle, distance, skill, in_sand):
    rot = np.array([[np.cos(angle), -np.sin(angle)], 
                    [np.sin(angle), np.cos(angle)]])
    rotated_location = np.matmul(rot, location[:,:,:,np.newaxis])
    rotated_location = rotated_location.squeeze(-1)
    translated_location = rotated_location - start

    angle_std = 1 / (2 * skill)
    dist_std = distance / skill

    if in_sand:
        angle_std *= 2
        dist_std *= 2
    
    rho = np.linalg.norm(rotated_location, axis=2)
    phi = np.arctan2(rotated_location[:,:,1], rotated_location[:,:,0])
    
    jacobian = distance
    return norm.pdf(rho, loc=distance, scale=dist_std) * norm.pdf(phi, loc=0, scale=angle_std) / jacobian

def transition_histogram_no_sample(start_x, start_y, skill, distance, angle, is_sand):
    transition = np.zeros((total_y_bins, total_x_bins))
    start = np.array([start_x, start_y])
    for xi in range(total_x_bins):
        for yi in range(total_y_bins):
            transition[yi, xi] = dblquad(lambda x, y: joint_density(np.array([[[x, y]]]), start, angle, distance, skill, is_sand), 
                                         y_bins[yi], y_bins[yi] + y_tick, 
                                         lambda _: x_bins[xi], lambda _: x_bins[xi] + x_tick)[0]
    return transition

def transition_histogram(start_x, start_y, skill, distance, angle, is_sand):
    H = sample_shots(x_bins, y_bins, start_x, start_y, skill, distance, angle, is_sand)

    transition = H.T
    start_xi, start_yi = to_bin(start_x, start_y)

    # print("start:", start_x, start_y, start_xi, start_yi)
    return_shots = 0
    for xi in range(total_x_bins):
        for yi in range(total_y_bins):
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


skill = 10
distance_levels = np.linspace(1, 200 + skill, dist_quant)
angle_levels = np.linspace(0, 2 * np.pi, angle_quant)

S = list(product(y_bins, x_bins))
A = list(product(distance_levels, angle_levels))
num_states = len(S)
num_actions = len(A)
print("states:", num_states)
print("actions:", num_actions)


def point_is_land(x, y):
    xi, yi = to_bin(x, y)
    return is_land[yi][xi]


def point_is_sand(x, y):
    xi, yi = to_bin(x, y)
    return is_sand[yi][xi]


unreachable_transition = np.array(
    [0 for _ in range(total_x_bins) for _ in range(total_y_bins)]
)
unreachable_transition[0] = 1


def transition_for_action_at_state(action, state):
    distance, angle = action
    y, x = state
    if not point_is_land(x, y):
        # print("Point is not on land")
        return unreachable_transition

    cx = x + 0.5 * x_tick
    cy = y + 0.5 * y_tick

    sand = point_is_sand(cx, cy)
    if not sand or distance <= (200 + skill) / 2:
        return transition_histogram(cx, cy, skill, distance, angle, sand).flatten()
    else:
        # print("In sand and distance is too long")
        # In sand, max distance is halved, so treat these actions as invalid
        return unreachable_transition


# Profiling how long generating T takes
# =====================================
#
# print(
#     "Time to generate all histograms for 1 action:",
#     timeit(gen_action_transitions, number=1),
# )

# "off", "pickle", "mmap"
T_cache = "pickle"
pickle_cache_file = "T.pkl"
mmap_cache_file = "T.npy"


def gen_action_transitions(action):
    return [transition_for_action_at_state(action, state) for state in S]


def mmap_gen_action_transitions(work):
    index, action = work
    T = np.memmap(
        mmap_cache_file,
        dtype="float",
        mode="r+",
        shape=(num_actions, num_states, num_states),
    )
    T[index] = [transition_for_action_at_state(action, state) for state in S]


def gen_T_parallel():
    t_start = perf_counter()
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    T = np.array(pool.map(gen_action_transitions, A))
    t_end = perf_counter()
    print("T generated in", t_end - t_start, "seconds")
    return T


if T_cache == "off":
    print("T caching is off, generating...")
    T = gen_T_parallel()
elif T_cache == "pickle":
    if exists(pickle_cache_file):
        print(f"Found cached {pickle_cache_file}")
        with open(pickle_cache_file, "rb") as f:
            T = pickle.load(f)
    else:
        print(f"No cached {pickle_cache_file}, generating...")
        T = gen_T_parallel()
        with open(pickle_cache_file, "wb") as f:
            pickle.dump(T, f)
elif T_cache == "mmap":
    has_cached = exists(mmap_cache_file)
    has_cached = exists(mmap_cache_file)
    T = np.memmap(
        mmap_cache_file,
        dtype="float",
        mode="r+" if has_cached else "w+",
        shape=(num_actions, num_states, num_states),
    )
    if not has_cached:
        print(f"No cached {mmap_cache_file}, generating...")
        t_start = perf_counter()
        pool = multiprocessing.Pool(multiprocessing.cpu_count())
        pool.map(mmap_gen_action_transitions, enumerate(A))
        t_end = perf_counter()
        print("T generated in", t_end - t_start, "seconds")
    else:
        print(f"Found cached {mmap_cache_file}")
else:
    assert T_cache in [
        "off",
        "pickle",
        "mmap",
    ], "T_cache must be one of: off, pickle, mmap"

R = np.array([-1 for _ in range(total_x_bins) for _ in range(total_y_bins)])
target_x, target_y = to_bin(*map["target"])
R[target_y * (total_x_bins) + target_x] = 1

plt.plot(x_bins[target_x], y_bins[target_y], "c.")

print("Training model...")
mdp = mdptoolbox.mdp.PolicyIteration(T, R, 0.99)
mdp.setVerbose()
mdp.run()
print("Converged in", mdp.time)

# Visualize values
# ================
#
draw_map()
draw_bins()

v_hist = np.split(np.array(mdp.V), total_y_bins)
X, Y = np.meshgrid(x_bins, y_bins)
plt.pcolormesh(X, Y, v_hist, alpha=0.8, vmin=90, vmax=100)

plt.savefig("value.png", dpi=400)

plt.clf()
plt.gca().invert_yaxis()
draw_map()
draw_bins()
# Visualize policy
policy_hist = np.split(np.array(mdp.policy), total_y_bins)
for yi in range(total_y_bins):
    for xi in range(total_x_bins):
        policy = policy_hist[yi][xi]
        if is_land[yi][xi]:
            distance, angle = A[policy_hist[yi][xi]]
            dx, dy = to_cartesian(distance, angle)
            start_x = x_bins[xi] + 0.5 * x_tick
            start_y = y_bins[yi] + 0.5 * y_tick
            plt.arrow(
                start_x,
                start_y,
                dx,
                dy,
                alpha=0.2,
                linewidth=1,
                head_width=8,
                head_length=8,
                length_includes_head=True,
            )
plt.savefig("policy.png", dpi=400)

# Visualize all shots
# ===================
#
# Convert into an animation with
# convert -delay 0 -loop 0 shots/*.png -quality 95 shots.mp4
#
# makedirs("shots", exist_ok=True)
# test_xi = 5
# test_yi = 13

# for ai, action in enumerate(A):
#     print("Rendering", ai + 1, "out of", len(A))
#     plt.clf()

#     plt.gca().set_xlim([-10, 800])
#     plt.gca().set_ylim([0, 600])
#     plt.gca().invert_yaxis()

#     draw_map()
#     draw_bins()

#     state_bin = test_yi * (total_x_bins) + test_xi
#     transitions = np.split(T[ai][state_bin], total_y_bins)
#     X, Y = np.meshgrid(x_bins, y_bins)
#     plt.pcolormesh(X, Y, transitions, alpha=0.8)

#     start_x = x_bins[test_xi] + 0.5 * x_tick
#     start_y = y_bins[test_yi] + 0.5 * y_tick

#     # Plot policy vector
#     policy = policy_hist[test_yi][test_xi]
#     distance, angle = A[policy_hist[test_yi][test_xi]]
#     dx, dy = to_cartesian(distance, angle)
#     plt.arrow(
#         start_x,
#         start_y,
#         dx,
#         dy,
#         color="green",
#         alpha=0.5,
#         linewidth=1,
#         head_width=8,
#         head_length=8,
#         length_includes_head=True,
#     )

#     # Plot this action
#     distance, angle = action
#     dx, dy = to_cartesian(distance, angle)
#     plt.arrow(
#         start_x,
#         start_y,
#         dx,
#         dy,
#         color="black",
#         alpha=0.5,
#         linewidth=1,
#         head_width=8,
#         head_length=8,
#         length_includes_head=True,
#     )

#     plt.title(f"Distance: {distance}, Angle: {angle}")

#     plt.savefig(f"shots/{ai}.png")
