from matplotlib import pyplot as plt
import csv
from pprint import pprint
from glob import glob
import re
from typing import Any
import os.path
from os import makedirs
from statistics import mean
import pdb

STRINGIFIED_KEYS = [
    "player_names",
    "skills",
    "scores",
    "player_states",
    "distances_from_target",
    "distance_source_to_target",
    "start",
    "target",
    "penalties",
    "timeout_count",
    "error_count",
    "winner_list",
    "total_time_sorted",
    "landing_history",
    "ending_history",
]

SINGLETON_ARRAY_KEYS = [
    "skills",
    "scores",
    "player_states",
    "distances_from_target",
    "penalties",
    "timeout_count",
    "error_count",
]

INT_KEYS = ["skills", "scores", "penalties", "timeout_count", "error_count"]

FLOAT_KEYS = ["distances_from_target", "distance_source_to_target"]

tournament_data = glob("*.csv")
file_pattern = re.compile(r"results_(?P<group>\d+)_(?P<part>\d+).csv")

# map -> skill -> group -> logs[]
logs = {}
total_lost_rows = 0
incomplete_rows = 0
for filename in tournament_data:
    match = file_pattern.match(filename)
    if not match:
        raise Exception(f"Couldn't match group number in {filename}")
    group = int(match.group("group"))
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for i, row in enumerate(reader):
            last_parse = None
            last_parse_key = None
            last_phase = None
            try:
                log: dict[Any, Any] = {
                    column: row[i] for i, column in enumerate(header)
                }
                # Parse JSON keys
                for key in STRINGIFIED_KEYS:
                    last_phase = "eval"
                    last_parse_key = key
                    last_parse = log[key]
                    log[key] = eval(log[key])

                # Extract singleton keys
                for key in SINGLETON_ARRAY_KEYS:
                    last_phase = "singletons"
                    last_parse_key = key
                    last_parse = log[key]
                    log[key] = log[key][0]

                for key in INT_KEYS:
                    last_phase = "ints"
                    last_parse_key = key
                    last_parse = log[key]
                    log[key] = int(log[key])

                for key in FLOAT_KEYS:
                    last_phase = "floats"
                    last_parse_key = key
                    last_parse = log[key]
                    log[key] = float(log[key])

                # Populate dictionary keys
                last_phase = "build"
                last_parse_key = None
                last_parse = None

                map = log["map"]
                skill = log["skills"]
                if not map in logs:
                    logs[map] = {}
                map_logs = logs[map]
                if not skill in map_logs:
                    map_logs[skill] = {}
                skill_logs = map_logs[skill]
                if not group in skill_logs:
                    skill_logs[group] = []
                group_logs = skill_logs[group]
                group_logs.append(log)
            except SyntaxError:
                print(f"Row {i} in {filename} is incomplete")
                incomplete_rows += 1
            except Exception as e:
                total_lost_rows += 1
                # Any exceptions are logged and skipped
                print(f"Failed to process row {i} in {filename}:")
                print(f"\t {repr(e)}")
                print(f"\t ({last_phase}) {last_parse_key}: {last_parse}")

print(f"Lost {total_lost_rows} rows while processing")
print(f"{incomplete_rows} were incomplete")
for map, map_logs in logs.items():
    for skill, skill_logs in map_logs.items():
        print(
            f"{map}, skill {skill}: {sum(len(group_logs) for group_logs in skill_logs.values())} logs"
        )

makedirs("plots", exist_ok=True)
for map, map_logs in logs.items():
    plt.clf()
    # Due to missing data, not all groups are always present
    groups = set()
    for skill_logs in map_logs.values():
        groups.update(skill_logs.keys())
    groups = list(groups)

    skill_levels = list(map_logs.keys())
    for group in groups:
        scores = []
        for skill_level in skill_levels:
            if group in map_logs[skill_level]:
                scores.append([log["scores"] for log in map_logs[skill_level][group]])
            else:
                scores.append(None)
        ys = [
            mean(skill_scores) if skill_scores is not None else None
            for skill_scores in scores
        ]
        if group == 5:
            print(map, ys)
        plt.plot(skill_levels, ys)
        # for i, point in list(enumerate(ys))[::-1]:
        #     if point is not None:
        #         plt.text(skill_levels[-1], ys[i], f"Group {group}")
        #         break
    plt.gca().legend([f"Group {group}" for group in groups])
    plt.savefig(f"plots/{os.path.basename(map)}.png")
