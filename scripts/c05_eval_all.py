"""Evaluate every checkpointed run under the network folder, on scene0.

One group of runs (the tag minus its seed) is one ``c04_eval`` call carrying
every seed of that group, so the summary keeps its across-seed mean +- std.
Everything else is c04_eval's own default: its episodes, mode and weights --
federated runs therefore use their server weights. Exactly one of

    --gt     ground truth, real stepping
    --virt   ground truth, virtual stepping
    --vis    images, real stepping

says how to evaluate *and* which runs it applies to: only tags carrying the same
``-gt``/``-virt`` markers are evaluated, so ``--virt`` covers the ``-gt-virt``
runs and ``--gt`` the ``-gt`` ones. Every run is scored on scene0, federated ones
included, since their tag names no scene. ``--container`` switches the scene to
ogbench's container-wide success instead of the strict one.

A run's checkpoints are ``ckp_*.pt`` in its own directory, or -- for a federated
run, whose clients keep their own states -- ``clients/global/ckp_*.pt``, the
shared server model.

    python -m scripts.c05_eval_all --virt
    python -m scripts.c05_eval_all --virt --container
    python -m scripts.c05_eval_all --vis --all-checkpoints --dry-run
"""

import argparse
import subprocess
import sys
from pathlib import Path

import conf.networks
from heca.misc import logger
from heca.misc.paths import data_root
from scripts.c04_eval import parse_run

#: The three ways to evaluate: (ground truth observations, virtual stepping).
MODES = {"gt": (True, False), "virt": (True, True), "vis": (False, False)}

#: Every evaluation is scored on scene0, also for federated runs whose tag names
#: no scene at all.
SCENE = "scene0"

#: Where a run keeps its checkpoints, relative to the run directory: its learner's
#: own, else the shared federated model every client of that run resumes from.
CHECKPOINTS = ("ckp_*.pt", "clients/global/ckp_*.pt")

DEFAULT_ROOT = data_root() / "network"


def by_step(path: Path) -> int:
    return int(path.stem.split("_")[1])


def experiment(name: str) -> str:
    """The run tag without its seed, which is what identifies one experiment."""
    stem, _, seed = name.rpartition("_s")
    return stem if stem and seed.isdigit() else name


def run_parts(name: str, default_scene: str) -> dict:
    """``parse_run`` for tags without a scene prefix, i.e. federated runs."""
    parts = parse_run(name)
    if parts["network"] is not None or "_" in experiment(name):
        return parts
    tokens = [token for token in experiment(name).split("-") if token]
    return {
        "scene": default_scene,
        "network": next((t for t in tokens if t in conf.networks.CONFIGS), None),
        "gt": "gt" in tokens,
        "virtual": "virt" in tokens,
        "seed": parts["seed"],
    }


def checkpoints_of(run_dir: Path) -> tuple[list[Path], str]:
    """This run's checkpoints, and the directory they sit in relative to it."""
    for pattern in CHECKPOINTS:
        found = sorted(run_dir.glob(pattern), key=by_step)
        if found:
            return found, pattern.rsplit("/", 1)[0] if "/" in pattern else ""
    return [], ""


def run_dirs(root: Path) -> list[Path]:
    """Every checkpointed run under ``root``: the folder's runs and its subfolders'."""
    found = [path for path in sorted(root.glob("*")) if path.is_dir()]
    found += [path for path in sorted(root.glob("*/*")) if path.is_dir()]
    return [path for path in found if checkpoints_of(path)[0]]


def steps_of(dirs: list[Path]) -> list[int]:
    """The checkpoint steps every run of a group has, so a step is comparable."""
    common = set.intersection(
        *({by_step(path) for path in checkpoints_of(run_dir)[0]} for run_dir in dirs)
    )
    for run_dir in dirs:
        missing = {by_step(p) for p in checkpoints_of(run_dir)[0]} - common
        if missing:
            logger.warning(f"{run_dir.name}: no checkpoint at step(s) {sorted(missing)}")
    return sorted(common)


def plan(
    root: Path,
    all_ckp: bool,
    gt: bool,
    virtual: bool,
    default_scene: str,
    only: list[str] | None = None,
) -> list:
    """One entry per group and checkpoint: (tags, parsed tag, relative ckp)."""
    groups: dict[str, list[Path]] = {}
    for run_dir in run_dirs(root):
        parts = run_parts(run_dir.name, default_scene)
        if parts["network"] is None:
            logger.warning(f"skip {run_dir.name}: the tag names no known network")
            continue
        if (parts["gt"], parts["virtual"]) != (gt, virtual):
            continue
        if only and not any(token in run_dir.name for token in only):
            continue
        groups.setdefault(experiment(run_dir.name), []).append(run_dir)

    jobs = []
    for dirs in groups.values():
        parts = run_parts(dirs[0].name, default_scene)
        names = [str(run_dir.relative_to(root)) for run_dir in dirs]
        _, prefix = checkpoints_of(dirs[0])
        steps = steps_of(dirs)
        if not steps:
            logger.warning(f"skip {dirs[0].name}: no checkpoint step shared by every seed")
            continue
        last = {
            run_dir.name: by_step(checkpoints_of(run_dir)[0][-1]) for run_dir in dirs
        }
        if len(set(last.values())) > 1:
            logger.warning(
                f"{dirs[0].name}: seeds end at different steps {last}, "
                f"evaluating the newest common one (ckp_{steps[-1]})"
            )
        chosen = [f"ckp_{step}.pt" for step in (steps if all_ckp else steps[-1:])]
        for name in chosen:
            jobs.append((names, parts, f"{prefix}/{name}" if prefix else name))
    return jobs


def command(root: Path, job: tuple, args, gt: bool, virtual: bool) -> list:
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
    cmd += ["--per-task", str(args.per_task), "--epochs", str(args.epochs)]
    if args.container:
        cmd.append("--container")
    if args.only:
        cmd += ["--only", *args.only]
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
    parser.add_argument(
        "--per-task", type=int, default=50, help="rollouts per task (default 50)"
    )
    parser.add_argument("--epochs", type=int, default=3, help="task sweeps (default 3)")
    parser.add_argument(
        "--only", nargs="+", default=None, help="keep runs whose name contains one of these"
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

    jobs = plan(args.root, args.all_checkpoints, gt, virtual, SCENE, args.only)
    if not jobs:
        raise SystemExit(f"no checkpointed run under {args.root} matches --{args.mode}")

    for index, job in enumerate(jobs, 1):
        names, parts, checkpoint = job
        logger.info(
            f"[{index}/{len(jobs)}] {args.mode}"
            f"{' container' if args.container else ''} / {parts['network']} / "
            f"{checkpoint} ({len(names)} seed(s))"
        )
        cmd = command(args.root, job, args, gt, virtual)
        if args.dry_run:
            print(" ".join(cmd))
        else:
            subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
