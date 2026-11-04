"""Evaluate every checkpointed run under the network folder, on scene0.

Each run is evaluated at its own newest checkpoint (the step of one seed is
never forced down to another seed's). Results are kept in one registry file per
(network, fed, eval mode, container):

    data/network/eval/{network}[_fed]_{mode}[_container].json

with one entry per run plus the across-seed aggregates (per observation
protocol: gt / gt-virt / vis). Re-running the sweep only evaluates runs whose
registry entry is missing or older than their newest checkpoint; ``--force``
re-evaluates everything. ``benchmark.json`` stays the compact per-configuration
aggregate, split by fed and container as well. The sweep ends with a per-run
checkpoint table, so runs that stopped early are visible at a glance.

Exactly one of ``--gt`` / ``--virt`` / ``--vis`` says which observation protocol
to evaluate (and which runs it applies to). ``--container`` switches the scene
to ogbench's container-wide success. Federated runs are detected by their
checkpoint location (``clients/global/ckp_*.pt``) and get their own files.

    python -m scripts.c05_eval_all --virt
    python -m scripts.c05_eval_all --virt --container
    python -m scripts.c05_eval_all --vis --force --dry-run
"""

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import conf.networks
from heca.misc import logger
from heca.misc.paths import data_root
from scripts.c04_eval import parse_run, save_benchmark, t_critical
from scripts.common.eval_registry import is_fresh, load_registry, registry_path, save_registry

#: The three observation protocols: (ground truth observations, virtual stepping).
MODES = {"gt": (True, False), "virt": (True, True), "vis": (False, False)}

#: Every evaluation is scored on scene0, also for federated runs whose tag names
#: name no scene at all.
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
    """This run's checkpoints (sorted by step), and their directory prefix."""
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


def plan(root, gt, virtual, scene, only, all_ckp) -> list[dict]:
    """One job per (run, checkpoint): the run's own newest step by default."""
    jobs: list[dict] = []
    for run_dir in run_dirs(root):
        parts = run_parts(run_dir.name, scene)
        if parts["network"] is None:
            logger.warning(f"skip {run_dir.name}: the tag names no known network")
            continue
        if (parts["gt"], parts["virtual"]) != (gt, virtual):
            continue
        if only and not any(token in run_dir.name for token in only):
            continue
        ckps, prefix = checkpoints_of(run_dir)
        if not ckps:
            logger.warning(f"skip {run_dir.name}: no checkpoint found")
            continue
        chosen = ckps if all_ckp else ckps[-1:]
        for path in chosen:
            jobs.append(
                {
                    "run_dir": run_dir,
                    "name": str(run_dir.relative_to(root)),
                    "parts": parts,
                    "ckp_rel": f"{prefix}/{path.name}" if prefix else path.name,
                    "fed": bool(prefix),
                }
            )
    return jobs


def command(root: Path, job: dict, args, gt: bool, virtual: bool) -> list:
    return [
        sys.executable,
        "-m",
        "scripts.c04_eval",
        "--root",
        str(root),
        "--tag",
        job["name"],
        "--network",
        job["parts"]["network"],
        "--ckp",
        job["ckp_rel"],
        "--scene",
        job["parts"]["scene"],
        "--mode",
        args.eval_mode,
        "--gt" if gt else "--no-gt",
        "--virtual" if virtual else "--no-virtual",
        "--per-task",
        str(args.per_task),
        "--epochs",
        str(args.epochs),
        "--no-summary",
    ] + (["--container"] if args.container else [])


def result_file(run_dir: Path, ckp_rel: str, mode: str, container: bool) -> Path:
    action = "_container" if container else ""
    return run_dir / "eval" / f"{Path(ckp_rel).stem}_{mode}{action}.json"


def merge_result(registry: dict, job: dict, result: dict) -> None:
    registry.setdefault("runs", {})[job["name"]] = {
        "checkpoint": job["ckp_rel"],
        "gt": job["parts"]["gt"],
        "virtual": job["parts"]["virtual"],
        "success_rate": result["success_rate"],
        "std_over_tasks": result.get("std_over_tasks", 0.0),
        "episodes": result.get("episodes"),
        "truncated": result.get("truncated"),
        "tasks": result.get("tasks", []),
        "timestamp": result.get("timestamp"),
    }


def update_benchmark(root, reg_path, network, fed, mode, container, scene) -> None:
    """Push each protocol aggregate of the registry into the benchmark table."""
    registry = load_registry(reg_path)
    for key, agg in registry.get("protocols", {}).items():
        rates = list(agg["success_rate_per_seed"].values())
        n = len(rates)
        episodes = [
            registry["runs"][name].get("episodes") or 0
            for name in agg["success_rate_per_seed"]
        ]
        episodes_per_seed = int(sum(episodes) / n) if n and any(episodes) else 0
        std = agg["std"]
        sem = std / (n**0.5) if n > 1 else 0.0
        ci = t_critical(n) * sem if n > 1 else 0.0
        save_benchmark(
            root,
            {
                "network": network,
                "scene": scene,
                "weights": "network",
                "gt": "gt" in key,
                "virtual": "virt" in key,
                "fed": fed,
                "mode": mode,
                "container": container,
                "greedy": True,
                "checkpoints": "per-run",
                "n_seeds": n,
                "tags": list(agg["success_rate_per_seed"].keys()),
                "success_rate_per_seed": rates,
                "episodes_per_seed": episodes_per_seed,
                "mean": agg["mean"],
                "std": std,
                "sem": sem,
                "ci95": ci,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            },
        )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument(
        "--all-checkpoints",
        action="store_true",
        help="evaluate every checkpoint step of each run instead of the newest one",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="re-evaluate runs whose registry entry is already up to date",
    )
    parser.add_argument(
        "--container", action="store_true", help="container-wide success"
    )
    parser.add_argument(
        "--eval-mode", default="task", help="c04_eval scene mode (default: task)"
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

    jobs = plan(args.root, gt, virtual, SCENE, args.only, args.all_checkpoints)
    if not jobs:
        raise SystemExit(f"no checkpointed run under {args.root} matches --{args.mode}")

    evaluated: list[tuple[str, str, str]] = []  # (run, checkpoint, status)
    for index, job in enumerate(jobs, 1):
        reg_path = registry_path(
            args.root, job["parts"]["network"], job["fed"], args.eval_mode, args.container
        )
        registry = load_registry(reg_path)
        result_path = result_file(
            job["run_dir"], job["ckp_rel"], args.eval_mode, args.container
        )
        if is_fresh(registry, job["name"], job["ckp_rel"]) and not args.force:
            logger.info(
                f"[{index}/{len(jobs)}] skip {job['name']} @ {job['ckp_rel']} "
                f"(already in {reg_path.relative_to(args.root)})"
            )
            evaluated.append((job["name"], job["ckp_rel"], "skip"))
            continue
        if result_path.exists() and not args.force:
            # a previous sweep (or a manual c04_eval) already wrote this result:
            # adopt it into the registry instead of re-running the episodes
            logger.info(
                f"[{index}/{len(jobs)}] adopt {job['name']} @ {job['ckp_rel']} "
                f"(existing {result_path.relative_to(job['run_dir'])})"
            )
            evaluated.append((job["name"], job["ckp_rel"], "adopted"))
            if args.dry_run:
                continue
            merge_result(registry, job, json.loads(result_path.read_text()))
            save_registry(reg_path, registry)
            continue

        logger.info(
            f"[{index}/{len(jobs)}] {args.mode}"
            f"{' container' if args.container else ''} / "
            f"{job['parts']['network']} / {job['name']} @ {job['ckp_rel']}"
        )
        cmd = command(args.root, job, args, gt, virtual)
        if args.dry_run:
            print(" ".join(cmd))
            continue
        subprocess.run(cmd, check=True)

        if not result_path.exists():
            logger.error(f"no result file written at {result_path}; not merged")
            evaluated.append((job["name"], job["ckp_rel"], "missing"))
            continue
        merge_result(registry, job, json.loads(result_path.read_text()))
        save_registry(reg_path, registry)
        evaluated.append((job["name"], job["ckp_rel"], "evaluated"))

    if args.dry_run:
        return

    # keep benchmark cells in sync with every registry the sweep touched
    for network in {job["parts"]["network"] for job in jobs}:
        for fed in (False, True):
            reg_path = registry_path(
                args.root, network, fed, args.eval_mode, args.container
            )
            if not reg_path.exists():
                continue
            update_benchmark(
                args.root, reg_path, network, fed, args.eval_mode, args.container, SCENE
            )

    # the per-run checkpoint table: early-stopped seeds are visible at a glance
    print("\n== per-run checkpoints ==")
    for name, ckp, status in evaluated:
        print(f"  {name:<40} {ckp:<30} {status}")


if __name__ == "__main__":
    main()
