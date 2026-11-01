"""Evaluate every checkpointed run under a root and collect the numbers in JSON.

For each run folder the highest ``ckp_<n>.pt`` is the one evaluated; every task of
the scene is rolled out ``--per-task`` times for ``--epochs`` sweeps, so the JSON
holds per-task success rates plus the combined mean and the spread over tasks --
enough to draw a table or a bar chart without re-running anything.

    python -m scripts.c04_eval_runs --root data/network/standard --out eval/runs.json
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import conf.networks
from heca.agents.heca import Heca
from heca.graphs.graph import SubgoalMode
from heca.learning.ppo import PPO
from heca.misc import hardware, logger
from heca.misc.paths import data_root
from scripts.c04_eval import find_checkpoint, parse_run, rollouts, weight_key
from scripts.common.helper import fmt_duration
from scripts.common.scenes import find_scene_models


def evaluate(heca: Heca, checkpoint: Path, per_task: int, epochs: int, seed: int) -> dict:
    weights = weight_key(torch.load(checkpoint, map_location=hardware.device,
                                    weights_only=False), "network")
    heca.learner.network.upgrade(weights)
    tasks = heca.scene.tasks
    if not tasks:
        raise RuntimeError("env exposes no task list")
    wins = [0] * len(tasks)
    counts = [0] * len(tasks)
    index = 0
    for _ in range(epochs):
        for idx in range(len(tasks)):
            heca.scene.task_id = idx + 1
            task_wins, _truncated = rollouts(heca, per_task, seed_from=seed + index)
            index += per_task
            wins[idx] += task_wins
            counts[idx] += per_task
    rates = [w / c for w, c in zip(wins, counts)]
    return {
        "tasks": [
            {
                "task_id": idx + 1,
                "task_name": str(task.get("task_name", idx + 1)),
                "success": wins[idx],
                "episodes": counts[idx],
                "success_rate": rates[idx],
            }
            for idx, task in enumerate(tasks)
        ],
        "success_rate": float(np.mean(rates)),
        "std_over_tasks": float(np.std(rates)),
        "episodes": int(sum(counts)),
    }


def runs_under(root: Path, only: list[str], skip: Path | None = None) -> list[Path]:
    """Every run folder that holds a checkpoint; ``skip`` drops the output folder."""
    found = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir()
        and not path.name.startswith(".")
        and path != skip
        and any(path.glob("ckp_*.pt"))
    ]
    if only:
        found = [path for path in found if any(token in path.name for token in only)]
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=data_root() / "network/standard")
    parser.add_argument("--out", type=Path, default=None,
                        help="default: <root>/eval/runs.json")
    parser.add_argument("--only", nargs="+", default=[],
                        help="keep runs whose name contains one of these words")
    parser.add_argument("--network", default="a0")
    parser.add_argument("--per-task", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out = args.out or args.root / "eval" / "runs.json"
    paths = runs_under(args.root, args.only, skip=out.parent)
    if not paths:
        raise SystemExit(f"no run with a checkpoint under {args.root}")
    out.parent.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    payload = json.loads(out.read_text()) if out.exists() else {"runs": []}
    known = {entry["tag"]: entry for entry in payload["runs"]}

    agents: dict[tuple, Heca] = {}
    started = time.perf_counter()
    for path in paths:
        parts = parse_run(path.name)
        checkpoint = find_checkpoint(path, "latest")
        key = (parts["scene"], parts["gt"], parts["virtual"])
        if key not in agents:
            scene, gt, virtual = key
            agents[key] = Heca.get(
                Heca.Config(
                    experts=find_scene_models(scene),
                    learner=PPO.Config(
                        tag=path.name,
                        network=conf.networks.get(parts["network"] or args.network),
                        wandb=logger.WandBConfig(enabled=False),
                        max_update=0,
                        lr_annealing=False,
                        normalize_rewards=False,
                    ),
                    inference=True,
                    virtual=virtual,
                    reload=False,
                    use_gt=gt,
                    smode=SubgoalMode.BOTH,
                )
            )
            agents[key].scene.set_mode("task")
        heca = agents[key]
        heca.learner.cfg.tag = path.name
        entry = {"tag": path.name, "checkpoint": checkpoint.name, "scene": key[0],
                 "network": parts["network"], "gt": key[1], "virtual": key[2],
                 "per_task": args.per_task, "epochs": args.epochs, "seed": args.seed,
                 "timestamp": datetime.now().isoformat(timespec="seconds")}
        try:
            entry.update(
                evaluate(heca, checkpoint, args.per_task, args.epochs, args.seed)
            )
        except Exception as error:  # a stale checkpoint must not end the sweep
            entry["error"] = f"{type(error).__name__}: {error}"
            logger.error(f"{path.name} / {checkpoint.name}: {entry['error']}")
        known[path.name] = entry
        payload["runs"] = sorted(known.values(), key=lambda item: item["tag"])
        payload["updated"] = entry["timestamp"]
        out.write_text(json.dumps(payload, indent=2) + "\n")
        if "success_rate" in entry:
            logger.info(
                f"{path.name} / {checkpoint.name}: {100 * entry['success_rate']:.1f}% "
                f"+- {100 * entry['std_over_tasks']:.1f} over {len(entry['tasks'])} tasks"
            )

    rates = [entry["success_rate"] for entry in payload["runs"] if "success_rate" in entry]
    payload["summary"] = {
        "n_runs": len(rates),
        "mean": float(np.mean(rates)) if rates else 0.0,
        "std": float(np.std(rates)) if rates else 0.0,
        "per_task": args.per_task,
        "epochs": args.epochs,
        "updated": datetime.now().isoformat(timespec="seconds"),
    }
    out.write_text(json.dumps(payload, indent=2) + "\n")
    if not rates:
        logger.error(f"no run evaluated successfully; see {out}")
        return
    logger.info(
        f"{len(rates)} run(s): {100 * np.mean(rates):.2f}% "
        f"+- {100 * np.std(rates):.2f} (SD over runs) -> {out} "
        f"in {fmt_duration(time.perf_counter() - started)}"
    )


if __name__ == "__main__":
    main()
