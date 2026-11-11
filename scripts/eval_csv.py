#!/usr/bin/env python3
"""Write the evaluation CSVs from the eval saves and the wandb curves.

Three files under <root>/eval:

  success_rates.csv   columns = networks, rows = sr, std, sr-container,
                      std-container (mean/std of the per-seed success rates)
  epochs_to_90.csv    columns = networks, rows = updates_to_90_mean,
                      updates_to_90_std (first update with >=90% smoothed
                      success, per training seed, from wandb)
  task_rates.csv      rows = networks, columns = tasks (mean per-task
                      success rate over the training seeds)

Per-run results are read from <root>/<run>/eval/ckp_*_task[_container].json
(the newest evaluated checkpoint per run); only the run dirs whose name
matches scene0_*-<network>-gt-virt_s* count.

    python -m scripts.eval_csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path
from statistics import mean, stdev

import numpy as np

from heca.misc.paths import data_root

NETWORKS = [f"a{i}" for i in range(0, 13)]  # a0 .. a12


def run_dirs(root: Path, network: str, only: list[str] | None = None) -> list[Path]:
    """Run dirs of one network; ``only`` keeps names containing one of the words."""
    dirs = sorted(
        p
        for p in (root / "standard").glob(f"scene0_*-{network}-gt-virt_s*")
        if p.is_dir()
    )
    if only:
        dirs = [p for p in dirs if any(token in p.name for token in only)]
    return dirs


def _step_of(path: Path) -> int | None:
    token = path.stem.split("_")[1] if len(path.stem.split("_")) > 1 else ""
    return int(token) if token.isdigit() else None


def newest_result(run_dir: Path, suffix: str, step: int | None = None) -> dict | None:
    """One per-run result file: the newest checkpoint, or the newest <= ``step``."""
    files = sorted(
        run_dir.glob(f"eval/ckp_*_task{suffix}.json"),
        key=lambda p: _step_of(p) if _step_of(p) is not None else -1,
    )
    if step is not None:
        files = [f for f in files if _step_of(f) is not None and _step_of(f) <= step]
    if not files:
        return None
    try:
        return json.loads(files[-1].read_text())
    except (json.JSONDecodeError, OSError):
        return None


def per_seed_rates(
    root: Path,
    network: str,
    container: bool,
    step: int | None = None,
    only: list[str] | None = None,
) -> list[float]:
    suffix = "_container" if container else ""
    rates = []
    for run_dir in run_dirs(root, network, only):
        payload = newest_result(run_dir, suffix, step)
        if payload and isinstance(payload.get("success_rate"), (int, float)):
            rates.append(float(payload["success_rate"]))
    return rates


def networks_of(root: Path) -> list[str]:
    """Networks that actually have run dirs (a0..a15, not the hardcoded list)."""
    found = []
    for path in sorted((root / "standard").glob("scene0_*")):
        for token in path.name.split("-"):
            if token.startswith("a") and token[1:].isdigit() and token not in found:
                found.append(token)
    return sorted(found, key=lambda n: int(n[1:])) or NETWORKS


def write_success_rates(
    root: Path, step: int | None = None, only: list[str] | None = None
) -> None:
    rows = {
        "sr": [],
        "std": [],
        "sr-container": [],
        "std-container": [],
    }
    nets = networks_of(root)
    for network in nets:
        plain = per_seed_rates(root, network, container=False, step=step, only=only)
        wide = per_seed_rates(root, network, container=True, step=step, only=only)
        rows["sr"].append(mean(plain) if plain else "")
        rows["std"].append(stdev(plain) if len(plain) > 1 else "")
        rows["sr-container"].append(mean(wide) if wide else "")
        rows["std-container"].append(stdev(wide) if len(wide) > 1 else "")
    path = root / "eval" / "success_rates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["network"] + nets)
        for name, values in rows.items():
            writer.writerow([name] + values)
    print(f"wrote {path}")


def write_task_rates(
    root: Path, step: int | None = None, only: list[str] | None = None
) -> None:
    tasks: list[str] = []
    matrix: dict[str, list[float]] = {}
    for network in networks_of(root):
        rates: list[list[float]] = []
        for run_dir in run_dirs(root, network, only):
            payload = newest_result(run_dir, "", step)
            if not payload or not isinstance(payload.get("tasks"), list):
                continue
            row = {}
            for task in payload["tasks"]:
                if isinstance(task.get("success_rate"), (int, float)):
                    row[str(task.get("task_name", task.get("task_id")))] = float(
                        task["success_rate"]
                    )
            if row:
                rates.append(row)
        if not rates:
            matrix[network] = []
            continue
        names = list(rates[0])
        tasks = names
        matrix[network] = [
            mean(r.get(name, float("nan")) for r in rates if name in r)
            for name in names
        ]
    path = root / "eval" / "task_rates.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["network"] + tasks)
        for network in NETWORKS:
            writer.writerow([network] + matrix.get(network, []))
    print(f"wrote {path}")


def updates_to_90() -> dict[str, list[float]]:
    """Per training seed: the first update with >=90% smoothed success."""
    import wandb

    from scripts.d01_plot_wandb_curves import load_run_series, merge_runs, smooth

    api = wandb.Api()
    config = wandb_config()
    result: dict[str, list[float]] = {}
    for network in NETWORKS:
        group = f"final-{network}-gt-virt"
        try:
            members = list(
                api.runs(
                    f"{config.entity}/{config.project}",
                    filters={"group": group},
                    per_page=50,
                )
            )
        except Exception as exc:  # pragma: no cover - api hiccup
            print(f"wandb: {exc}", file=sys.stderr)
            return result
        by_name: dict[str, list] = {}
        for run in members:
            by_name.setdefault(str(run.name), []).append(run)
        hits: list[float] = []
        for name, runs_of_name in by_name.items():
            pieces = [
                loaded
                for run in runs_of_name
                if (loaded := load_run_series(run, "stats/success_rate", 10000))
                is not None
            ]
            if not pieces:
                continue
            steps, values = merge_runs(pieces)
            smoothed = smooth(values, 20)
            idx = int(np.argmax(smoothed >= 0.9)) if (smoothed >= 0.9).any() else -1
            if idx >= 0:
                hits.append(float(steps[idx]))
            print(f"  {group}/{name}: 90% at update {hits[-1] if hits else 'never'}")
        result[network] = hits
    return result


def wandb_config():
    from heca.misc.logger import WandBConfig

    return WandBConfig()


def write_epochs_to_90(root: Path, hits: dict[str, list[float]]) -> None:
    path = root / "eval" / "epochs_to_90.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["network"] + NETWORKS)
        writer.writerow(
            ["updates_to_90_mean"]
            + [mean(hits[n]) if hits.get(n) else "" for n in NETWORKS]
        )
        writer.writerow(
            ["updates_to_90_std"]
            + [
                stdev(hits[n]) if len(hits.get(n, [])) > 1 else ""
                for n in NETWORKS
            ]
        )
    print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root", type=Path, default=data_root() / "network",
        help="directory holding the run dirs (default: <data>/network)",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        metavar="N",
        help="Build the tables from the newest evaluation with step <= N (one row "
        "per network, i.e. a single 750-step evaluation) instead of the newest "
        "result on disk.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help="Keep only run dirs whose name contains one of these words, e.g. "
        "'--only final' to match the sweep's --only.",
    )
    args = parser.parse_args()

    write_success_rates(args.root, args.step, args.only)
    write_task_rates(args.root, args.step, args.only)
    hits = updates_to_90()
    if hits:
        write_epochs_to_90(args.root, hits)
    else:
        print("wandb: no 90%-crossing data, epochs_to_90.csv skipped", file=sys.stderr)


if __name__ == "__main__":
    main()
