#!/usr/bin/env python3
"""Bucket the per-epoch success counts (0..N) across every eval save, per task.

Every task-mode evaluation saves, for each run and task, how many of its
``--per-task`` episodes succeeded in each of the ``--epochs`` blocks
(``wins_per_epoch`` / ``success_rate_per_epoch``). This script pools those
blocks over all runs (registries and per-run result files alike) and reports,
per task: the total number of blocks (epochs), and for each possible count
0..N how many blocks ended with that many successes plus its percentage.

    python eval_prob
    python eval_prob --include a0 a1 a2
    python eval_prob --exclude 'scene0_final-a12-gt-virt_s*'
    python eval_prob --both          # plain and container protocols in one go
    python eval_prob --json --out eval_prob.json

Results are saved as CSV to ``<root>/eval/eval_prob[_container].csv`` (one
column per task, rows 0..N = the success count, values = the bucket's share);
the per-task table is printed to stdout. ``--json`` prints the full JSON
buckets, ``--out`` writes it to a file.

``--include`` keeps only the named networks (exact network tokens, e.g. a0;
a0 never matches a10). Run names for --exclude are matched as shell patterns
against the run dir basename (e.g. ``scene0_final-a12-gt-virt_s2``), so seeds
or whole networks can be dropped. When both are given, --exclude wins.
"""

import argparse
import csv
import fnmatch
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

from heca.misc.paths import data_root


def network_of(run_name: str) -> str | None:
    """The network token of a run dir name (``scene0_final-a0-gt-virt_s0`` -> a0)."""
    stem = Path(run_name).name
    stem = stem.rpartition("_s")[0] or stem
    rest = stem.partition("_")[2]
    for token in rest.split("-"):
        if re.fullmatch(r"[ax]\d+", token):
            return token
    return None


def _read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return None


def collect(root: Path, container: bool) -> list[dict]:
    """Every evaluated run: name, checkpoint and its per-task epoch wins.

    Sources (deduplicated by run basename + checkpoint):
      * the eval registries (``<...>/eval/<network>_task[_container].json``),
      * the per-run result files (``<run dir>/eval/ckp_*_task[_container].json``).
    """
    seen: set[tuple[str, str]] = set()
    entries: list[dict] = []

    def add(run_name: str, checkpoint: str, tasks: list[dict]) -> None:
        key = (Path(run_name).name, str(Path(str(checkpoint)).stem))
        if key in seen:
            return
        seen.add(key)
        entries.append(
            {
                "run": Path(run_name).name,
                "checkpoint": str(checkpoint),
                "tasks": [
                    t for t in tasks if isinstance(t, dict) and t.get("wins_per_epoch")
                ],
            }
        )

    suffix = "_container" if container else ""
    # 1) registry files: <root>/eval/... and <root>/*/eval/...
    for rel in (f"eval/*_task{suffix}.json", f"*/eval/*_task{suffix}.json"):
        for path in sorted(root.glob(rel)):
            payload = _read_json(path)
            if not payload or path.name == "benchmark.json":
                continue
            runs = payload.get("runs")
            if isinstance(runs, dict):
                for name, entry in runs.items():
                    if isinstance(entry, dict):
                        add(name, entry.get("checkpoint", ""), entry.get("tasks", []))
            # old-style per-run result files lying in an eval dir
            tasks = payload.get("tasks")
            if isinstance(tasks, list):
                add(payload.get("tag", path.name), payload.get("checkpoint", ""), tasks)

    # 2) per-run result files next to the checkpoints
    for path in sorted(root.glob(f"*/*/eval/ckp_*_task{suffix}.json")):
        payload = _read_json(path)
        if payload:
            add(
                payload.get("tag", path.parent.parent.name),
                payload.get("checkpoint", ""),
                payload.get("tasks", []),
            )

    return entries


def analyze(
    root: Path, include: list[str], exclude: list[str], container: bool
) -> dict | None:
    """Bucket every kept eval save; prints the per-task table, returns JSON.

    Returns ``None`` when the protocol has no eval saves at all (so ``--both``
    can skip it); raises when saves exist but none carry per-epoch data.
    """
    entries = collect(root, container)
    if not entries:
        print(
            f"note: no {'container ' if container else ''}eval saves under {root}",
            file=sys.stderr,
        )
        return None
    if include:
        included = set(include)
        entries = [e for e in entries if network_of(e["run"]) in included]
        seen = {network_of(e["run"]) for e in entries}
        for name in sorted(included - seen):
            print(
                f"note: --include {name} matched no "
                f"{'container ' if container else ''}runs",
                file=sys.stderr,
            )
        if not entries:
            return None
    kept = []
    skipped_runs = 0
    for entry in entries:
        if any(fnmatch.fnmatch(entry["run"], pattern) for pattern in exclude):
            continue
        if entry["tasks"]:
            kept.append(entry)
        else:
            skipped_runs += 1
    if not kept:
        raise SystemExit(
            f"no eval saves with per-epoch data under {root} "
            f"({skipped_runs} run(s) without it; they were evaluated before the "
            "per-epoch fields existed -- re-evaluate to fill the buckets)"
        )
    if skipped_runs:
        print(
            f"note: {skipped_runs} run(s) had no per-epoch data and were skipped",
            file=sys.stderr,
        )

    # task key -> {block_size, buckets, total}
    buckets: dict[tuple, dict] = defaultdict(
        lambda: {"block_size": 0, "counts": defaultdict(int), "total": 0}
    )
    for entry in kept:
        for task in entry["tasks"]:
            key = (str(task.get("task_name", "")), task.get("task_id"))
            cell = buckets[key]
            block = int(task.get("episodes_per_epoch") or 0)
            cell["block_size"] = max(cell["block_size"], block)
            for wins in task["wins_per_epoch"]:
                cell["counts"][int(wins)] += 1
                cell["total"] += 1

    result = {"container": container, "tasks": []}
    print(
        f"--- {'container' if container else 'plain'} protocol ---\n"
        f"{'task':<28} {'id':>3} {'blocks':>6} {'block':>5} {'mean':>8}   "
        f"buckets (count: n (%))"
    )
    for (name, task_id), cell in sorted(buckets.items()):
        total = cell["total"]
        size = cell["block_size"]
        counts = cell["counts"]
        mean = sum(k * v for k, v in counts.items()) / total if total else 0.0
        nonzero = sorted(counts.items())
        result["tasks"].append(
            {
                "task_name": name,
                "task_id": task_id,
                "episodes_per_epoch": size,
                "total_epochs": total,
                "mean_successes": round(mean, 3),
                "mean_success_rate": round(mean / size, 4) if size else 0.0,
                "bucket_counts": {str(k): counts.get(k, 0) for k in range(size + 1)},
                "bucket_percentages": {
                    str(k): round(100.0 * counts.get(k, 0) / total, 3)
                    for k in range(size + 1)
                },
            }
        )
        line = f"{name:<28} {str(task_id):>3} {total:>6} {size:>5} {100 * mean / size:>7.1f}%"
        compact = "  ".join(f"{k}:{v} ({100 * v / total:.0f}%)" for k, v in nonzero)
        print(f"{line}   {compact}")
    return result


def write_csv(path: Path, result: dict) -> None:
    """One column per task: the bucket index (0..N) and each bucket's rate.

    First column is ``x`` (the success count), every task column holds the
    fraction of that task's epochs that ended with exactly that many successes.
    """
    tasks = result["tasks"]
    if not tasks:
        return
    n = max(int(t["episodes_per_epoch"]) for t in tasks)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x"] + [str(t["task_name"]) for t in tasks])
        for k in range(n + 1):
            row = [k]
            for task in tasks:
                total = int(task["total_epochs"])
                count = int(task["bucket_counts"].get(str(k), 0))
                row.append(f"{count / total:.4f}" if total else "0")
            writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=data_root() / "network",
        help="directory holding the run dirs and the eval/ registries "
        "(default: <data>/network).",
    )
    parser.add_argument(
        "--include",
        nargs="+",
        default=[],
        help="network names to keep, e.g. 'a0 a1 a2' (exact network tokens; "
        "a0 never matches a10).",
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=[],
        help="run names (shell patterns) to skip, e.g. "
        "scene0_final-a12-gt-virt_s2 scene0_final-a12-gt-virt_s1",
    )
    parser.add_argument(
        "--container",
        action="store_true",
        help="use the *_task_container.json saves instead of the plain task ones.",
    )
    parser.add_argument(
        "--both",
        action="store_true",
        help="bucket the plain and the container protocol in one run; each is "
        "saved to its own file.",
    )
    parser.add_argument(
        "--json", action="store_true", help="also print the full JSON buckets."
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="optional: also write the JSON buckets here.",
    )
    args = parser.parse_args()

    if args.both and args.out:
        parser.error(
            "--out cannot be combined with --both; the two protocols get "
            "their own default files"
        )
    modes = [False, True] if args.both else [args.container]

    combined: dict[str, dict] = {}
    for container in modes:
        suffix = "_container" if container else ""
        result = analyze(args.root, args.include, args.exclude, container)
        if result is None:
            continue  # the note was already printed
        combined[suffix.lstrip("_") or "plain"] = result
        csv_path = args.root / "eval" / f"eval_prob{suffix}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_csv(csv_path, result)
        print(f"wrote {csv_path}", file=sys.stderr)
        if args.out and not args.both:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(result, indent=2) + "\n")
            print(f"wrote {args.out}", file=sys.stderr)
        if not args.both and args.json:
            print(json.dumps(result, indent=2))
    if not combined:
        raise SystemExit(f"no eval saves with per-epoch data under {args.root}")
    if args.both and args.json:
        print(json.dumps(combined, indent=2))


if __name__ == "__main__":
    main()
