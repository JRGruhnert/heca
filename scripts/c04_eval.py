import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")

from heca.agents.heca import Heca
from heca.graphs.graph import SubgoalMode
from heca.learning.ppo import PPO
from heca.misc import hardware, logger
from scripts.common.helper import fmt_duration
from scripts.common.scenes import find_scene_models

import conf.networks

RUN_ROOT = Path("data/network/standard")


def find_checkpoint(run_dir: Path, name: str) -> Path:
    if name != "latest":
        return run_dir / name
    checkpoints = sorted(
        run_dir.glob("ckp_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"no ckp_*.pt in {run_dir}")
    return checkpoints[-1]


def rollouts(heca: Heca, count: int, seed_from: int) -> tuple[int, int]:
    wins = truncated = 0
    for i in range(count):
        heca.scene.seed = seed_from + i
        x, y = heca.sample()
        _, fb = heca.act(x, y)
        wins += int(fb.success)
        truncated += int(fb.truncated)
    return wins, truncated


def t_critical(n: int) -> float:
    """97.5% quantile of Student's t with ``n-1`` dof (95% CI of the mean)."""
    if n <= 1:
        return 1.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96  # normal approximation fallback


def save_summary(*, network: str, payload: dict) -> Path:
    """Write the across-seed summary JSON for one network (paper table row)."""
    out_dir = RUN_ROOT / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    action = "" if payload.get("greedy", True) else "_sampled"
    path = out_dir / f"{network}_{payload['mode']}{action}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"  saved {path}")
    return path


def resolve_tags(patterns: list[str]) -> list[str]:
    """Expand ``--tag`` entries: exact run dir names or globs under RUN_ROOT."""
    tags: list[str] = []
    for pattern in patterns:
        if any(ch in pattern for ch in "*?["):
            found = sorted(p.name for p in RUN_ROOT.glob(pattern) if p.is_dir())
            if not found:
                raise FileNotFoundError(f"{pattern!r} matches no run dir in {RUN_ROOT}")
        else:
            if not (RUN_ROOT / pattern).is_dir():
                raise FileNotFoundError(f"no run dir {RUN_ROOT / pattern}")
            found = [pattern]
        for name in found:
            if name not in tags:
                tags.append(name)
    return tags


def weight_key(checkpoint: dict, weights: str) -> dict:
    """The state dict to evaluate, and a clear error if Ditto weights are absent."""
    key = "personal_network" if weights == "personal" else "network"
    if key not in checkpoint:
        raise KeyError(
            f"checkpoint has no {key!r}: it was written by a run without "
            f"--method ditto (keys: {sorted(k for k in checkpoint if k.endswith('network'))})"
        )
    return checkpoint[key]


def save_result(run_dir: Path, payload: dict) -> Path:
    out_dir = run_dir / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    action = "" if payload.get("greedy", True) else "_sampled"
    stem = Path(str(payload["checkpoint"])).stem
    path = out_dir / f"{stem}_{payload['mode']}{action}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"  saved {path}")
    return path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tag",
        nargs="+",
        required=True,
        help="run dir(s) under data/network/standard: one per trained seed. "
        "Shell patterns work too, e.g. 'scene0_final-a0-gt-virt_s*'.",
    )
    ap.add_argument("--ckp", default="latest", help="file name, or 'latest'")
    ap.add_argument("--network", default="a0", help="config name from conf.networks")
    ap.add_argument("--scene", default="scene0")
    ap.add_argument(
        "--per-task",
        type=int,
        default=50,
        help="rollouts per task (paper: 50; their script's default is 20)",
    )
    ap.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="task sweeps to average (the paper's 3 evaluation epochs)",
    )
    ap.add_argument(
        "--episodes", type=int, default=250, help="episodes, randomized mode"
    )
    ap.add_argument(
        "--seed", type=int, default=0, help="env RNG seed, shared by every tag"
    )
    ap.add_argument(
        "--mode",
        default="task",
        help="scene env mode: 'task' (eval protocol, one task at a time) or "
        "'randomized' (the training distribution)",
    )
    ap.add_argument(
        "--weights",
        choices=("network", "personal"),
        default="network",
        help="which weights to evaluate: the federated model, or the Ditto "
        "personal model (personal_network) when the run used --method ditto",
    )
    ap.add_argument("--gt", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--virtual", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument(
        "--sample",
        action="store_true",
        help="sample options instead of taking the argmax, i.e. act the way "
        "training does; the default (greedy) matches OGBench's eval_temperature 0",
    )
    args = ap.parse_args()
    args.tag = resolve_tags(args.tag)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    heca = Heca.get(
        Heca.Config(
            agents=find_scene_models(args.scene),
            learner=PPO.Config(
                tag=args.tag[0],
                network=conf.networks.get(args.network),
                wandb=logger.WandBConfig(enabled=False),
                max_update=0,
                lr_annealing=False,
                # keep fb.reward raw: success is read straight off it
                normalize_rewards=False,
            ),
            visualize=False,
            inference=True,
            virtual=args.virtual,
            reload=False,
            use_gt=args.gt,
            smode=SubgoalMode.BOTH,
        )
    )

    heca.scene.set_mode(args.mode)
    logger.info(
        f"{len(args.tag)} tag(s), net={args.network}, mode={heca.scene.cfg.mode}, "
        f"greedy={not args.sample}"
    )

    started = time.perf_counter()

    if args.mode != "task":
        per_tag: list[float] = []
        for tag in args.tag:
            path = find_checkpoint(RUN_ROOT / tag, args.ckp)
            checkpoint = torch.load(
                path, map_location=hardware.device, weights_only=False
            )
            missing, unexpected = heca.learner.network.upgrade(
                weight_key(checkpoint, args.weights)
            )
            if missing or unexpected:
                logger.warning(
                    f"  {tag}: checkpoint mismatch, {len(missing)} missing, "
                    f"{len(unexpected)} unexpected parameter(s)"
                )
            wins, truncated = rollouts(heca, args.episodes, seed_from=args.seed)
            rate = wins / args.episodes
            per_tag.append(rate)
            logger.info(
                f"{tag} / {path.name}: randomized episodes, success "
                f"{wins}/{args.episodes} = {100 * rate:.1f}% ({truncated} "
                f"truncated) in {fmt_duration(time.perf_counter() - started)}"
            )
            save_result(
                RUN_ROOT / tag,
                {
                    "tag": tag,
                    "checkpoint": path.name,
                    "scene": args.scene,
                    "network": args.network,
                "weights": args.weights,
                    "mode": args.mode,
                    "greedy": not args.sample,
                    "episodes": args.episodes,
                    "seed": args.seed,
                    "success": wins,
                    "success_rate": rate,
                    "truncated": truncated,
                    "checkpoint_missing": len(missing),
                    "checkpoint_unexpected": len(unexpected),
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                },
            )

        spread = stdev(per_tag) if len(per_tag) > 1 else 0.0
        sem = spread / (len(per_tag) ** 0.5) if len(per_tag) > 1 else 0.0
        ci = t_critical(len(per_tag)) * sem if len(per_tag) > 1 else 0.0
        logger.info(
            f"across {len(per_tag)} tag(s): {100 * mean(per_tag):.2f}% "
            f"± {100 * spread:.2f} (1 SD over seeds)"
            + (f", 95% CI ± {100 * ci:.2f}" if len(per_tag) > 1 else "")
        )
        save_summary(
            network=args.network,
            payload={
                "network": args.network,
                "weights": args.weights,
                "scene": args.scene,
                "mode": args.mode,
                "greedy": not args.sample,
                "checkpoints": args.ckp,
                "episodes_per_seed": args.episodes,
                "episode_seed": args.seed,
                "n_seeds": len(per_tag),
                "tags": args.tag,
                "success_rate_per_seed": per_tag,
                "mean": mean(per_tag),
                "std": spread,
                "sem": sem,
                "ci95": ci,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )
        return

    tasks = heca.scene.tasks
    if not tasks:
        raise RuntimeError(
            f"env of {args.scene} exposes no task list; use --mode randomized"
        )

    per_tag: list[float] = []
    for tag in args.tag:
        path = find_checkpoint(RUN_ROOT / tag, args.ckp)
        checkpoint = torch.load(path, map_location=hardware.device, weights_only=False)
        missing, unexpected = heca.learner.network.upgrade(
            weight_key(checkpoint, args.weights)
        )
        if missing or unexpected:
            logger.warning(
                f"  {tag}: checkpoint mismatch, {len(missing)} missing, "
                f"{len(unexpected)} unexpected parameter(s)"
            )

        wins = [0] * len(tasks)
        counts = [0] * len(tasks)
        truncated_total = 0
        index = 0
        for _epoch in range(args.epochs):
            for idx in range(len(tasks)):
                heca.scene.task_id = idx + 1
                task_wins, task_truncated = rollouts(
                    heca, args.per_task, seed_from=args.seed + index
                )
                index += args.per_task
                wins[idx] += task_wins
                counts[idx] += args.per_task
                truncated_total += task_truncated

        rates = [w / c for w, c in zip(wins, counts)]
        overall = mean(rates)
        per_tag.append(overall)
        table = "\n".join(
            f"  {idx + 1}. {str(task.get('task_name', idx + 1)):<24} "
            f"{wins[idx]:>4}/{counts[idx]:<4} = {100 * rates[idx]:5.1f}%"
            for idx, task in enumerate(tasks)
        )
        logger.info(
            f"{tag} / {path.name} "
            f"({args.per_task} rollouts/task x {args.epochs} epoch(s), "
            f"{'sampled' if args.sample else 'greedy'})\n"
            f"{table}\n"
            f"  {tag}: overall {100 * overall:.1f}% "
            f"(unweighted mean over {len(rates)} tasks, "
            f"{sum(counts)} episodes, {truncated_total} truncated)"
        )
        save_result(
            RUN_ROOT / tag,
            {
                "tag": tag,
                "checkpoint": path.name,
                "scene": args.scene,
                "network": args.network,
                "weights": args.weights,
                "mode": args.mode,
                "greedy": not args.sample,
                "per_task": args.per_task,
                "epochs": args.epochs,
                "seed": args.seed,
                "success_rate": overall,
                "std_over_tasks": stdev(rates) if len(rates) > 1 else 0.0,
                "episodes": sum(counts),
                "truncated": truncated_total,
                "checkpoint_missing": len(missing),
                "checkpoint_unexpected": len(unexpected),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
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
            },
        )

    spread = stdev(per_tag) if len(per_tag) > 1 else 0.0
    sem = spread / (len(per_tag) ** 0.5) if len(per_tag) > 1 else 0.0
    ci = t_critical(len(per_tag)) * sem if len(per_tag) > 1 else 0.0
    logger.info(
        f"across {len(per_tag)} tag(s): {100 * mean(per_tag):.2f}% "
        f"± {100 * spread:.2f} (1 SD over seeds)"
        + (f", 95% CI ± {100 * ci:.2f}" if len(per_tag) > 1 else "")
        + f" in {fmt_duration(time.perf_counter() - started)}"
    )
    save_summary(
        network=args.network,
        payload={
            "network": args.network,
            "scene": args.scene,
            "mode": args.mode,
            "greedy": not args.sample,
            "checkpoints": args.ckp,
            "per_task": args.per_task,
            "epochs": args.epochs,
            "episodes_per_seed": sum(counts),
            "episode_seed": args.seed,
            "n_seeds": len(per_tag),
            "tags": args.tag,
            "success_rate_per_seed": per_tag,
            "mean": mean(per_tag),
            "std": spread,
            "sem": sem,
            "ci95": ci,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
        },
    )


if __name__ == "__main__":
    main()
