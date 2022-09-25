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
import pdb


# =====================
# Problem Configuration
# =====================
#
x_quant = 20
y_quant = 10
# Distance rating: 200 + s
dist_quant = 20
angle_quant = 36
skill = 10

# map_file = "maps/default/simple_with_sandtraps.json"
map_file = "maps/g5/zigzag_g5.json"
# map_file = "maps/g5/yiwei_wu_map.json"
with open(map_file) as f:
    map = json.loads(f.read())

# =============================
# Map quantization fundamentals
# =============================
#
map_xs = [x for x, _ in map["map"]]
map_ys = [y for _, y in map["map"]]

map_min_x = min(map_xs)
map_max_x = max(map_xs)
map_min_y = min(map_ys)
map_max_y = max(map_ys)

print(f"Map boundaries: x {map_min_x}, {map_max_x} y {map_min_y} {map_max_y}")

x_tick = (map_max_x - map_min_x) / x_quant
y_tick = (map_max_y - map_min_y) / y_quant

min_x = map_min_x - x_tick
max_x = map_max_x + x_tick
min_y = map_min_y - y_tick
max_y = map_max_y + y_tick

print(f"Bin boundaries: x {min_x}, {max_x} y {min_y} {max_y}")

width = max_x - min_x
height = max_y - min_y

x_bins = np.linspace(min_x, max_x, x_quant + 2, endpoint=False)
y_bins = np.linspace(min_y, max_y, y_quant + 2, endpoint=False)
total_x_bins = len(x_bins)
total_y_bins = len(y_bins)
print("bins:", total_x_bins, total_y_bins)

# ==============
# Geometry setup
# ==============
#
green_poly = ShapelyPolygon(map["map"])
sand_polys = [ShapelyPolygon(coords) for coords in map["sand traps"]]

distance_levels = np.linspace(1, 200 + skill, dist_quant)
angle_levels = np.linspace(0, 2 * np.pi, angle_quant)

cell_polys = [
    [
        ShapelyPolygon(
            [
                (x_bin, y_bin),
                (x_bin + x_tick, y_bin),
                (x_bin + x_tick, y_bin + y_tick),
                (x_bin, y_bin + y_tick),
                (x_bin, y_bin),
            ]
        )
        for x_bin in x_bins
    ]
    for y_bin in y_bins
]

# =============
# Terrain types
# =============
#
is_land = [[green_poly.contains(cell_poly) for cell_poly in row] for row in cell_polys]
has_land = [
    [green_poly.intersects(cell_poly) for cell_poly in row] for row in cell_polys
]
percent_land = [
    [
        min(green_poly.intersection(cell_poly).area / cell_poly.area, 1)
        for cell_poly in row
    ]
    for row in cell_polys
]
percent_sand = [
    [
        min(
            sum(sand_poly.intersection(cell_poly).area for sand_poly in sand_polys)
            / cell_poly.area,
            1,
        )
        for cell_poly in row
    ]
    for row in cell_polys
]


def to_bin_index(x, y):
    # xi = int((x - min_x) / (x_tick))
    # yi = int((y - min_y) / (y_tick))
    xi = next(xi for xi, x_bin in enumerate(x_bins) if x_bin > x) - 1
    yi = next(yi for yi, y_bin in enumerate(y_bins) if y_bin > y) - 1
    return xi, yi


# =====================
# MDP: States & Actions
# =====================
#
landy_bins = list(
    (yi, xi)
    for yi, xi in product(range(total_y_bins), range(total_x_bins))
    if has_land[yi][xi]
)
S = list(
    (yi, xi, terrain)
    for ((yi, xi), terrain) in product(landy_bins, ["green", "sand"])
    if (
        (terrain == "green" and percent_sand[yi][xi] < 1)
        or (terrain == "sand" and percent_sand[yi][xi] > 0)
    )
)
# Dead state for invalid moves
S.append((None, None, None))
# (xi, yi) -> index in S
S_index = {(xi, yi, terrain): index for index, (yi, xi, terrain) in enumerate(S)}
A = list(product(distance_levels, angle_levels))
num_states = len(S)
num_actions = len(A)
print("states:", num_states)
print("actions:", num_actions)


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


# =====================
# Visualization Helpers
# =====================
#
def draw_bins():
    for x in x_bins:
        plt.axvline(x=x, color="black", alpha=0.1)
    for y in y_bins:
        plt.axhline(y=y, color="black", alpha=0.1)


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


def reset_figure():
    plt.clf()
    plt.gca().invert_yaxis()
    # plt.gca().set_aspect(width / height)
    draw_bins()
    draw_map()


def overlay_tiles(tiles, vmin=None, vmax=None):
    X, Y = np.meshgrid(x_bins + 0.5 * x_tick, y_bins + 0.5 * y_tick)
    plt.pcolormesh(X, Y, tiles, alpha=0.5, vmin=vmin, vmax=vmax)


# ===========================
# Shot transition calculation
# ===========================
#
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

    dist_rv = norm(loc=distance, scale=distance_dev)
    angle_rv = norm(loc=angle, scale=angle_dev)
    ds = dist_rv.rvs(size=num_samples)
    # Naive rolling distance
    ds *= 1.1
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


def joint_density(location, start, angle, distance, skill, in_sand):
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    rotated_location = np.matmul(rot, location[:, :, :, np.newaxis])
    rotated_location = rotated_location.squeeze(-1)
    translated_location = rotated_location - start

    angle_std = 1 / (2 * skill)
    dist_std = distance / skill

    if in_sand:
        angle_std *= 2
        dist_std *= 2

    rho = np.linalg.norm(translated_location, axis=2)
    phi = np.arctan2(translated_location[:, :, 1], translated_location[:, :, 0])

    jacobian = distance
    return (
        norm.pdf(rho, loc=distance, scale=dist_std)
        * norm.pdf(phi, loc=0, scale=angle_std)
        / jacobian
    )


def transition_histogram_no_sample(start_x, start_y, skill, distance, angle, is_sand):
    transition = np.zeros((total_y_bins, total_x_bins))
    start = np.array([start_x, start_y])
    for xi in range(total_x_bins):
        for yi in range(total_y_bins):
            transition[yi, xi] = dblquad(
                lambda x, y: joint_density(
                    np.array([[[x, y]]]), start, angle, distance, skill, is_sand
                ),
                y_bins[yi],
                y_bins[yi] + y_tick,
                lambda _: x_bins[xi],
                lambda _: x_bins[xi] + x_tick,
            )[0]
    return transition


def transition_histogram(start_xi, start_yi, skill, distance, angle, is_sand):
    start_x = x_bins[start_xi] + 0.5 * x_tick
    start_y = y_bins[start_yi] + 0.5 * y_tick
    H = sample_shots(x_bins, y_bins, start_x, start_y, skill, distance, angle, is_sand)

    transition = np.zeros(num_states)

    return_shots = 0
    for xi in range(total_x_bins):
        for yi in range(total_y_bins):
            samples_in_bin = H.T[yi][xi]
            p_land = percent_land[yi][xi]
            p_sand = percent_sand[yi][xi]

            grounded_samples = p_land * samples_in_bin
            drowned_samples = (1 - p_land) * samples_in_bin
            return_shots += drowned_samples

            sandy_samples = p_sand * grounded_samples
            green_samples = (1 - p_sand) * grounded_samples
            sand_state = (xi, yi, "sand")
            green_state = (xi, yi, "green")
            if sand_state in S_index:
                transition[S_index[sand_state]] += sandy_samples
            elif sandy_samples > 0:
                print("Lost some sand samples", xi, yi, sandy_samples)

            if green_state in S_index:
                transition[S_index[green_state]] += green_samples
            elif green_samples > 0:
                print("Lost some green samples", xi, yi, green_samples)

    # Allocate returned shots to the start state
    start_key = (start_xi, start_yi, "sand" if is_sand else "green")
    if not start_key in S_index:
        print("Mismatch between states and index?")
        pdb.set_trace()
    start_i = S_index[start_key]
    transition[start_i] += return_shots

    if np.abs(np.sum(H) - np.sum(transition)) > 0.001:
        print(
            f"Mis-allocated samples from {start_x},{start_y}: had {np.sum(H)}, finished with {np.sum(transition)}."
        )

    # Normalize to get probabilities
    transition = transition / max(np.sum(transition), 1)

    return transition


# Used for invalid actions
# S[-1] is an "invalid" sink state: shots going there stay there forever
unreachable_transition = np.array([0 for _ in range(num_states)])
unreachable_transition[-1] = 1


def transition_for_action_at_state(action, state):
    distance, angle = action
    yi, xi, terrain = state
    # This should be a no-op since there should be no non-land states
    if (
        # Not a no-op because we need a dead state
        state == (None, None, None)
        # No-ops, can probably be deleted
        or not has_land[yi][xi]
        or (terrain == "green" and percent_sand[yi][xi] == 1)
        or (terrain == "sand" and percent_sand[yi][xi] == 0)
    ):
        return unreachable_transition

    if terrain == "green":
        return transition_histogram(xi, yi, skill, distance, angle, False).flatten()
    elif distance <= (200 + skill) / 2:
        return transition_histogram(xi, yi, skill, distance, angle, True).flatten()
    else:
        # In sand, max distance is halved, so treat these actions as invalid
        return unreachable_transition


# ==============================
# Actual T calculation & storage
# ==============================
#
# This can be: "off", "pickle", "mmap"
# Use "mmap" when T is too large to fit in memory, but it will slow down all
# calculations by a LOT (> 2x).
T_cache = "off"
pickle_cache_file = "T.pkl"
mmap_cache_file = "T.npy"


def gen_action_transitions(action):
    return [transition_for_action_at_state(action, state) for state in S]


# =====================================
# Profiling how long generating T takes
# =====================================
#
# print(
#     "Time to generate all histograms for 1 action:",
#     timeit(lambda: gen_action_transitions((50, 0)), number=1),
# )


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
    # T = np.array([gen_action_transitions(action) for action in A])
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

# =============
# Reward vector
# =============
#
# All rewards are -1 (penalty for taking another shot) except for the target
# which has the only positive reward, incentivizing the MDP solving algorithm
# to find quick paths to reach the target.
R = np.array([-1 for _ in range(num_states)])
target_xi, target_yi = to_bin_index(*map["target"])
ti = S_index[(target_xi, target_yi, "green")]
R[ti] = 1

# ===========
# Train model
# ===========
#
print("Training model...")
mdp = mdptoolbox.mdp.PolicyIteration(T, R, 0.89, max_iter=20)
mdp.setVerbose()
mdp.run()
print("Converged in", mdp.time)

# ================
# Visualize values
# ================
#
v_hist = np.zeros((total_y_bins, total_x_bins))
sorted_values = sorted(mdp.V[:-1])
vmin = sorted_values[1]
vmax = sorted_values[-1]
for i, value in enumerate(mdp.V[:-1]):
    yi, xi, terrain = S[i]
    if terrain == "green":
        v_hist[yi][xi] = value
    elif terrain == "sand" and percent_sand[yi][xi] > 0.9:
        v_hist[yi][xi] = value
reset_figure()
overlay_tiles(v_hist, vmin=vmin, vmax=vmax)
plt.title(f"Values: {map_file}, skill {skill}")
plt.savefig("value.png", dpi=400)

# ================
# Visualize policy
# ================
#
reset_figure()
for i, policy in enumerate(mdp.policy[:-1]):
    yi, xi, terrain = S[i]
    distance, angle = A[policy]
    dx, dy = to_cartesian(distance, angle)
    start_x = x_bins[xi] + 0.5 * x_tick
    start_y = y_bins[yi] + 0.5 * y_tick
    plt.arrow(
        start_x,
        start_y,
        dx,
        dy,
        color="black" if terrain == "green" else "red",
        alpha=0.2,
        linewidth=1,
        head_width=8,
        head_length=8,
        length_includes_head=True,
    )
plt.title(f"Policy: {map_file}, skill {skill}")
plt.savefig("policy.png", dpi=400)

# ===================
# Visualize all shots
# ===================
#
# Convert into an animation with
# `convert -delay 0 -loop 0 shots/*.png -quality 95 shots.mp4`
#
# makedirs("shots", exist_ok=True)
# test_xi = 3
# test_yi = 14
# test_terrain = "green"

# for ai, action in enumerate(A):
#     print("Rendering", ai + 1, "out of", len(A))

#     reset_figure()
#     plt.gca().set_xlim([min_x, max_x])
#     plt.gca().set_ylim([min_y, max_y])
#     plt.gca().invert_yaxis()

#     draw_map()
#     draw_bins()

#     state_bin = S_index[(test_xi, test_yi, test_terrain)]
#     transitions = T[ai][state_bin][:-1]
#     transition_probs = np.zeros((total_y_bins, total_x_bins))
#     for si, prob in enumerate(transitions):
#         xi, yi, terrain = S[si]
#         transition_probs[yi][xi] = prob
#     overlay_tiles(transition_probs, vmin=0, vmax=1)

#     start_x = x_bins[test_xi] + 0.5 * x_tick
#     start_y = y_bins[test_yi] + 0.5 * y_tick

#     # Plot policy vector
#     policy = mdp.policy[state_bin]
#     distance, angle = A[policy]
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
