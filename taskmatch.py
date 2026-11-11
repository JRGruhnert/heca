#!/usr/bin/env python3
"""Which ogbench tasks could a randomized start/goal have produced? (JSON out)

Samples ``--episodes`` randomized episodes (ground truth) and, per task,
``--task-episodes`` episodes in task mode, then measures for every entity and
axis:

* the randomized distribution (mean / std / min / max) and the task one;
* ``z`` -- how far the task's fixed value sits from the randomized cloud, in
  randomized standard deviations. ``|z| > ~2`` means the task value is outside
  the randomized range, so no randomized episode can reproduce it;
* match fractions of the randomized episodes against each task (start, goal,
  both, and "at least one task"), for several acceptance criteria;
* the effector yaw, per mode.

Everything goes into one JSON so the plots can be made later without
re-sampling:

    python taskmatch.py --episodes 10000 --task-episodes 500 \
        --out task_match_scene0.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from heca.misc import logger
from heca.misc.paths import find_repo_root
from heca.scenes.scene import Scene

from scripts.common.scenes import scene_config

#: acceptance criteria: "sigma" multiplies the task's own std, "abs" is metres
CRITERIA = (
    ("3sigma", 3.0, None),
    ("abs_0.01", None, 0.01),
    ("abs_0.02", None, 0.02),
    ("abs_0.05", None, 0.05),
)


def collect(scene, episodes: int, seed_from: int, task_id: int | None) -> dict:
    """Per-entity metric positions (start, goal) and the effector yaw."""
    scene.task_id = task_id
    start, goal, yaw = {}, {}, []
    start_state, goal_state = {}, {}
    for i in range(episodes):
        scene.seed = seed_from + i
        (x, _), (y, _) = scene.sample_task(with_scenes=True)
        for label in scene.entities:
            value_x = np.asarray(x[label].value, dtype=float)
            value_y = np.asarray(y[label].value, dtype=float)
            start.setdefault(label, []).append(
                scene.unnormalize_position(value_x[:3])
            )
            goal.setdefault(label, []).append(
                scene.unnormalize_position(value_y[:3])
            )
            start_state.setdefault(label, []).append([float(np.ravel(value_x)[-1])])
            goal_state.setdefault(label, []).append([float(np.ravel(value_y)[-1])])
        try:
            yaw.append(float(np.ravel(scene.last_rot)[0]))
        except (AttributeError, TypeError):
            yaw.append(float("nan"))
    return {
        "start": {k: np.asarray(v) for k, v in start.items()},
        "goal": {k: np.asarray(v) for k, v in goal.items()},
        "start_state": {k: np.asarray(v) for k, v in start_state.items()},
        "goal_state": {k: np.asarray(v) for k, v in goal_state.items()},
        "yaw": np.asarray(yaw),
    }


def describe(values: np.ndarray) -> dict:
    return {
        "mean": values.mean(axis=0).round(6).tolist(),
        "std": values.std(axis=0).round(6).tolist(),
        "min": values.min(axis=0).round(6).tolist(),
        "max": values.max(axis=0).round(6).tolist(),
        "n": int(len(values)),
    }


def fixed_entities(random: dict, task: dict, factor: float = 0.25) -> list[str]:
    """Entities whose spread in task mode is much smaller than randomized."""
    out = []
    for label, values in task["start"].items():
        spread_task = float(values.std(axis=0).mean())
        spread_random = float(random["start"][label].std(axis=0).mean())
        if spread_random > 1e-6 and spread_task < factor * spread_random:
            out.append(label)
    return out


def state_pinned(random: dict, task: dict) -> list[str]:
    """Entities whose *state* the task fixes (the buttons: position never moves)."""
    out = []
    for label, values in task["start_state"].items():
        spread = float(values.std(axis=0).mean())
        spread_random = float(random["start_state"][label].std(axis=0).mean())
        if spread_random > 1e-9 and spread < 0.25 * spread_random:
            out.append(label)
    return out


def state_masks(random: dict, task: dict, labels: list[str]) -> tuple:
    """Masks: does a randomized episode carry the task's required state?"""
    hit_start = np.ones(len(random["yaw"]), dtype=bool)
    hit_goal = np.ones(len(random["yaw"]), dtype=bool)
    for label in labels:
        for side, hits in (("start_state", hit_start), ("goal_state", hit_goal)):
            wanted = float(np.round(task[side][label].mean(), 0))
            hits &= np.abs(random[side][label][:, 0] - wanted) <= 0.5
    return hit_start, hit_goal


def match_masks(random: dict, task: dict, labels: list[str], sigma, tolerance):
    """Masks: does a randomized episode match this task's start / goal?"""
    hit_start = np.ones(len(random["yaw"]), dtype=bool)
    hit_goal = np.ones(len(random["yaw"]), dtype=bool)
    if not labels:
        return hit_start, hit_goal
    for label in labels:
        for side, hits in (("start", hit_start), ("goal", hit_goal)):
            mean = task[side][label].mean(axis=0)
            if tolerance is not None:
                width = np.full(mean.shape, float(tolerance))
            else:
                width = sigma * np.maximum(task[side][label].std(axis=0), 1e-6)
            hits &= np.all(np.abs(random[side][label] - mean) <= width, axis=1)
    state_start, state_goal = state_masks(random, task, state_pinned(random, task))
    return hit_start & state_start, hit_goal & state_goal


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--task-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="task_match_scene0.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    if not out_path.is_absolute():
        out_path = find_repo_root() / out_path

    scene = Scene.get(scene_config(args.scene))
    scene.set_mode("randomized")
    logger.info(f"randomized: {args.episodes} episodes")
    random = collect(scene, args.episodes, args.seed, None)

    scene.set_mode("task")
    tasks = list(range(1, len(scene.tasks) + 1))
    logger.info(f"task mode: {len(tasks)} task(s), {args.task_episodes} each")

    payload = {
        "scene": args.scene,
        "episodes": args.episodes,
        "task_episodes": args.task_episodes,
        "seed": args.seed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "task_info_keys": sorted(scene.tasks[0]) if scene.tasks else [],
        "criteria": [
            {"name": name, "sigma": sigma, "abs_m": tol}
            for name, sigma, tol in CRITERIA
        ],
        "randomized": {
            "yaw_std": float(np.nanstd(random["yaw"])),
            "yaw_range": [
                float(np.nanmin(random["yaw"])),
                float(np.nanmax(random["yaw"])),
            ],
            "start": {k: describe(v) for k, v in random["start"].items()},
            "goal": {k: describe(v) for k, v in random["goal"].items()},
            "start_state": {
                k: describe(v) for k, v in random["start_state"].items()
            },
            "goal_state": {k: describe(v) for k, v in random["goal_state"].items()},
        },
        "tasks": {},
        "union": {},
    }

    unions = {
        name: [np.zeros(len(random["yaw"]), bool), np.zeros(len(random["yaw"]), bool)]
        for name, _, _ in CRITERIA
    }
    for task_id in tasks:
        task = collect(scene, args.task_episodes, args.seed, task_id)
        labels = fixed_entities(random, task)
        entry = {
            "task_name": str(scene.tasks[task_id - 1].get("task_name", task_id)),
            "fixed_entities": labels,
            "state_entities": state_pinned(random, task),
            "start": {k: describe(v) for k, v in task["start"].items()},
            "goal": {k: describe(v) for k, v in task["goal"].items()},
            "start_state": {
                k: describe(v) for k, v in task["start_state"].items()
            },
            "goal_state": {k: describe(v) for k, v in task["goal_state"].items()},
            "z_offset_start": {
                label: [
                    float(
                        (
                            task["start"][label].mean(axis=0)[axis]
                            - random["start"][label].mean(axis=0)[axis]
                        )
                        / max(float(random["start"][label].std(axis=0)[axis]), 1e-9)
                    )
                    for axis in range(3)
                ]
                for label in scene.entities
            },
            "yaw_std": float(np.nanstd(task["yaw"])),
            "yaw_range": [
                float(np.nanmin(task["yaw"])),
                float(np.nanmax(task["yaw"])),
            ],
            "matches": {},
        }
        pos_start, pos_goal = match_masks(random, task, labels, 3.0, None)
        sta_start, sta_goal = state_masks(random, task, entry["state_entities"])
        entry["position_only"] = {
            "start": float(pos_start.mean()),
            "goal": float(pos_goal.mean()),
        }
        entry["state_only"] = {
            "start": float(sta_start.mean()),
            "goal": float(sta_goal.mean()),
        }
        for name, sigma, tol in CRITERIA:
            hit_start, hit_goal = match_masks(random, task, labels, sigma, tol)
            unions[name][0] |= hit_start
            unions[name][1] |= hit_goal
            entry["matches"][name] = {
                "start": float(hit_start.mean()),
                "goal": float(hit_goal.mean()),
                "both": float((hit_start & hit_goal).mean()),
            }
        payload["tasks"][str(task_id)] = entry
        logger.info(
            f"task {task_id} ({entry['task_name']}): "
            + "  ".join(
                f"{name} start={100 * entry['matches'][name]['start']:.2f}%"
                f" goal={100 * entry['matches'][name]['goal']:.2f}%"
                f" both={100 * entry['matches'][name]['both']:.2f}%"
                for name, _, _ in CRITERIA
            )
        )

    for name, (start_mask, goal_mask) in unions.items():
        payload["union"][name] = {
            "start": float(start_mask.mean()),
            "goal": float(goal_mask.mean()),
            "both": float((start_mask & goal_mask).mean()),
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"wrote {out_path}")
    for name, values in payload["union"].items():
        print(
            f"any task [{name}]: start {100 * values['start']:.2f}%  "
            f"goal {100 * values['goal']:.2f}%  both {100 * values['both']:.2f}%"
        )


if __name__ == "__main__":
    main()
