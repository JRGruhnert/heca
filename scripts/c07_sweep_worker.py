"""Sweep worker: evaluate every checkpoint of one run under one protocol.

Writes one JSON line per checkpoint (append mode) to
``<out>/raw/<tag>__<protocol>.jsonl`` so an interrupted sweep can resume:
checkpoints already present in the file are skipped.

Protocols:
  virt-container : gt experts, virtual stepping, container-wide success
  virt-real      : gt experts, virtual stepping, strict success
  gt-real        : gt experts, real stepping, strict success

Usage:
    python -m scripts.c07_sweep_worker --run-dir DIR --tag NAME \
        --network a12 --protocol virt-container --last 750 --out eval/
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from statistics import mean

import conf.networks
from heca.agents.heca import Heca
from heca.graphs.graph import SubgoalMode
from heca.learning.ppo import PPO
from heca.misc import hardware, logger
from scripts.c04_eval import rollouts, weight_key
from scripts.common.helper import fmt_duration
from scripts.common.scenes import find_scene_models

PROTOCOLS = ("virt-container", "virt-real", "gt-real")


def checkpoints(ckp_dir: Path, last: int, only: int | None = None) -> list[tuple[int, Path]]:
    found = []
    for path in sorted(ckp_dir.glob("ckp_*.pt"), key=lambda p: int(p.stem.split("_")[1])):
        epoch = int(path.stem.split("_")[1])
        if last and epoch > last:
            continue
        if only is not None and epoch != only:
            continue
        found.append((epoch, path))
    return found


def evaluate(
    heca: Heca,
    checkpoint: Path,
    per_task: int,
    epochs: int,
    seed: int,
) -> dict:
    weights = weight_key(
        torch.load(checkpoint, map_location=hardware.device, weights_only=False),
        "network",
    )
    missing, unexpected = heca.learner.network.upgrade(weights)
    if missing or unexpected:
        logger.warning(
            f"  checkpoint mismatch: {len(missing)} missing, "
            f"{len(unexpected)} unexpected parameter(s)"
        )
    tasks = heca.scene.tasks
    if not tasks:
        raise RuntimeError("env exposes no task list")
    wins = [0] * len(tasks)
    counts = [0] * len(tasks)
    truncated = 0
    index = 0
    for _ in range(epochs):
        for idx in range(len(tasks)):
            heca.scene.task_id = idx + 1
            task_wins, task_truncated = rollouts(heca, per_task, seed_from=seed + index)
            index += per_task
            wins[idx] += task_wins
            counts[idx] += per_task
            truncated += task_truncated
    rates = [w / c for w, c in zip(wins, counts)]
    return {
        "success_rate": float(mean(rates)),
        "std_over_tasks": float(np.std(rates)),
        "episodes": int(sum(counts)),
        "truncated": int(truncated),
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
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run-dir", type=Path, required=True)
    ap.add_argument("--ckp-dir", type=Path, default=None)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--network", required=True)
    ap.add_argument("--protocol", choices=PROTOCOLS, required=True)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--last", type=int, default=0, help="max epoch, 0 = all")
    ap.add_argument("--epoch", type=int, default=None, help="exactly this epoch")
    ap.add_argument("--per-task", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("eval"))
    args = ap.parse_args()

    ckp_dir = args.ckp_dir or args.run_dir
    epochs = checkpoints(ckp_dir, args.last, args.epoch)
    if not epochs:
        logger.warning(f"{args.tag} [{args.protocol}]: no checkpoints in {ckp_dir}")
        return

    raw_dir = args.out / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{args.tag}__{args.protocol}.jsonl"
    done: set[int] = set()
    if raw_path.exists():
        for line in raw_path.read_text().splitlines():
            try:
                done.add(int(json.loads(line)["epoch"]))
            except Exception:
                continue
    todo = [(e, p) for e, p in epochs if e not in done]
    if not todo:
        logger.info(f"{args.tag} [{args.protocol}]: all {len(epochs)} checkpoints done")
        return

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    experts = find_scene_models(args.scene)
    container = args.protocol == "virt-container"
    virtual = args.protocol != "gt-real"
    for expert in experts:
        expert.scene.container_wide_success = container

    heca = Heca.get(
        Heca.Config(
            experts=experts,
            learner=PPO.Config(
                tag=args.tag,
                network=conf.networks.get(args.network),
                wandb=logger.WandBConfig(enabled=False),
                max_update=0,
                lr_annealing=False,
                normalize_rewards=False,
            ),
            inference=True,
            virtual=virtual,
            reload=False,
            use_gt=True,
            smode=SubgoalMode.BOTH,
        )
    )
    heca.scene.set_mode("task")

    started = time.perf_counter()
    for idx, (epoch, path) in enumerate(todo, start=1):
        result = evaluate(heca, path, args.per_task, args.epochs, args.seed)
        payload = {
            "tag": args.tag,
            "protocol": args.protocol,
            "epoch": epoch,
            "checkpoint": path.name,
            "success_rate": result["success_rate"],
            "std_over_tasks": result["std_over_tasks"],
            "episodes": result["episodes"],
            "truncated": result["truncated"],
            "tasks": result["tasks"],
            "per_task": args.per_task,
            "epochs": args.epochs,
            "seed": args.seed,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        }
        with raw_path.open("a") as fh:
            fh.write(json.dumps(payload) + "\n")
        logger.info(
            f"{args.tag} [{args.protocol}] ckp_{epoch}: "
            f"{100 * result['success_rate']:.1f}% "
            f"({idx}/{len(todo)} in {fmt_duration(time.perf_counter() - started)})"
        )


if __name__ == "__main__":
    main()
