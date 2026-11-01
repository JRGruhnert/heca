"""Evaluate every checkpointed run under a root, with the flags its tag names.

One group of runs (the tag minus its seed) is one ``c04_eval`` call carrying
every seed of that group, so the summary keeps its across-seed mean +- std.
Everything else is c04_eval's own default: its episodes, mode and weights --
federated runs therefore use their server weights. Exactly one of

    --gt     ground truth, real stepping
    --virt   ground truth, virtual stepping
    --vis    images, real stepping

says how to evaluate, and ``--container`` switches the scene to ogbench's
container-wide success instead of the strict one.

    python -m scripts.c05_eval_all --virt
    python -m scripts.c05_eval_all --virt --container
    python -m scripts.c05_eval_all --vis --all-checkpoints --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

from heca.misc import logger
from scripts.c04_eval import DEFAULT_ROOT, parse_run

#: The three ways to evaluate: (ground truth observations, virtual stepping).
MODES = {"gt": (True, False), "virt": (True, True), "vis": (False, False)}


def by_step(path: Path) -> int:
    return int(path.stem.split("_")[1])


def experiment(name: str) -> str:
    """The run tag without its seed, which is what identifies one experiment."""
    stem, _, seed = name.rpartition("_s")
    return stem if stem and seed.isdigit() else name


def steps_of(run_dirs: list[Path]) -> list[int]:
    """The checkpoint steps every run of a group has, so a step is comparable."""
    common = set.intersection(
        *({by_step(path) for path in run_dir.glob("ckp_*.pt")} for run_dir in run_dirs)
    )
    for run_dir in run_dirs:
        missing = {by_step(p) for p in run_dir.glob("ckp_*.pt")} - common
        if missing:
            logger.warning(f"{run_dir.name}: no checkpoint at step(s) {sorted(missing)}")
    return sorted(common)


def plan(root: Path, all_ckp: bool) -> list:
    """One entry per group and checkpoint: (tags, parsed tag, ckp)."""
    groups: dict[str, list[Path]] = {}
    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        if not any(run_dir.glob("ckp_*.pt")):
            continue
        if parse_run(run_dir.name)["network"] is None:
            logger.warning(f"skip {run_dir.name}: the tag names no known network")
            continue
        groups.setdefault(experiment(run_dir.name), []).append(run_dir)

    jobs = []
    for run_dirs in groups.values():
        parts = parse_run(run_dirs[0].name)
        names = [run_dir.name for run_dir in run_dirs]
        checkpoints = ["latest"] if not all_ckp else [
            f"ckp_{step}.pt" for step in steps_of(run_dirs)
        ]
        jobs += [(names, parts, checkpoint) for checkpoint in checkpoints]
    return jobs


def command(root: Path, job: tuple, gt: bool, virtual: bool, container: bool) -> list:
    names, parts, checkpoint = job
    cmd = [
        sys.executable,
        "-m",
        "scripts.c04_eval",
        "--root",
        str(root),
        "--tag",
        *names,
        "--network",
        parts["network"],
        "--ckp",
        checkpoint,
        "--scene",
        parts["scene"],
        "--gt" if gt else "--no-gt",
        "--virtual" if virtual else "--no-virtual",
    ]
    if container:
        cmd.append("--container")
    return cmd


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="evaluate every common checkpoint step instead of the last one",
    )
    parser.add_argument(
        "--container", action="store_true", help="container-wide success"
    )
    parser.add_argument("--dry-run", action="store_true")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--gt", dest="mode", action="store_const", const="gt", help="ground truth, real stepping"
    )
    mode.add_argument(
        "--virt",
        dest="mode",
        action="store_const",
        const="virt",
        help="ground truth, virtual stepping",
    )
    mode.add_argument(
        "--vis", dest="mode", action="store_const", const="vis", help="images, real stepping"
    )
    args = parser.parse_args()
    gt, virtual = MODES[args.mode]

    jobs = plan(args.root, args.all_checkpoints)
    if not jobs:
        raise SystemExit(f"no checkpointed run under {args.root}")

    for index, job in enumerate(jobs, 1):
        names, parts, checkpoint = job
        logger.info(
            f"[{index}/{len(jobs)}] {args.mode}"
            f"{' container' if args.container else ''} / {parts['network']} / "
            f"{checkpoint} ({len(names)} seed(s))"
        )
        cmd = command(args.root, job, gt, virtual, args.container)
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
