"""Write one CSV per wandb run, for the LaTeX ``trainingsplot`` package.

Unlike ``d01_plot_wandb_curves`` this does no aggregation and draws nothing: it
dumps every run on its own, with **all** of its metrics as columns, so
``trainingsplot.sty`` can average or compare the files itself.

Same bookkeeping as the plotter, minus the averaging:

* a run's *group* is its wandb group, else its name without the ``_s<seed>``
  suffix, and ``--group`` accepts the same shell patterns (``'final-a*-gt-virt'``);
* runs that share a name (a crash and its restart) are merged into one series --
  the run with the most data wins on overlapping updates and the others fill
  its gaps;
* seeds stay separate: every name becomes its own CSV.

Layout of ``--out`` (default ``<repo or $HECA_OUT>/csv``)::

    <out>/<group>/<run-name>.csv     # header: update, env, <metrics...>
    <out>/<group>/index.json         # files of the group + per-run metadata
    <out>/index.json                 # every group at once
    <out>/trainingsplot_groups.txt   # one "group=a.csv,b.csv" line per group

Column names are the wandb metric keys with ``/`` and any other non
alphanumeric run turned into one separator, e.g. ``stats/success_rate`` ->
``stats.success_rate``, because the CSV separator is the comma and the LaTeX
macro takes the column name as a plain argument.  ``update`` is the wandb step
and ``env`` is ``step * capacity`` (the ``x=env`` axis of the plotter).

Within a group the files line up row by row -- the LaTeX macro averages by row
index and takes the x values from the first file -- so ``trainingsplot_groups.txt``
lists the longest run first; with ``update``/``env`` as the x column that is the
only ordering that keeps the x values exact.

    python -m scripts.d03_wandb_csv --group 'final-a*-gt-virt' \
        --out csv
"""

import argparse
import csv
import fnmatch
import json
import re
import sys
from pathlib import Path

import numpy as np

from heca.misc.paths import output_root

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from heca.misc.logger import WandBConfig  # noqa: E402

try:
    import wandb
except ModuleNotFoundError as exc:  # pragma: no cover - environment check
    raise SystemExit(f"wandb is not installed in the active environment: {exc}") from exc


DEFAULT_CAPACITY = 2048  # option steps per learner update (buffer capacity)


# --------------------------------------------------------------------------- #
# run discovery (same rules as d01_plot_wandb_curves)
# --------------------------------------------------------------------------- #
def group_of(run) -> str | None:
    """Group a run belongs to: its wandb group, else its name without the seed."""
    group = getattr(run, "group", None)
    if group:
        return str(group)
    stripped = re.sub(r"_s\d+$", "", str(run.name))
    return stripped or None


def match_groups(runs, patterns: list[str]) -> dict[str, list]:
    """``{group: [run, ...]}`` for every group matching one of ``patterns``."""
    matched: dict[str, list] = {}
    for run in runs:
        group = group_of(run)
        if group is None:
            continue
        if any(fnmatch.fnmatch(group, pattern) for pattern in patterns):
            matched.setdefault(group, []).append(run)
    return dict(sorted(matched.items()))


def _regex_any(patterns: list[str]) -> str:
    """Mongo-style regex matching any of the shell patterns (server-side filter)."""
    return "|".join(
        fnmatch.translate(p).removeprefix("(?s:").removesuffix(")\\Z")
        for p in patterns
    )


def capacity_of(run, fallback: int) -> int:
    """Buffer capacity logged with the run (option steps between two updates)."""
    config = getattr(run, "config", None) or {}
    for key in ("buffer/capacity", "buffer.capacity", "capacity"):
        value = config.get(key) if hasattr(config, "get") else None
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return fallback


# --------------------------------------------------------------------------- #
# one run -> (steps, {metric: values})
# --------------------------------------------------------------------------- #
def fetch_run(run, samples: int) -> tuple[np.ndarray, dict[str, np.ndarray]] | None:
    """Every numeric metric of a run, keyed by wandb step."""
    try:
        hist = run.history(samples=samples)
    except Exception:
        return None
    if hist is None or hist.empty or "_step" not in hist.columns:
        return None

    steps = hist["_step"].to_numpy(dtype=float)
    order = np.argsort(steps)
    steps = steps[order]

    metrics: dict[str, np.ndarray] = {}
    for column in hist.columns:
        if str(column).startswith("_"):
            continue
        try:
            values = hist[column].to_numpy(dtype=float)[order]
        except (TypeError, ValueError):
            continue  # not a numeric metric
        if not np.isfinite(values).any():
            continue
        metrics[str(column)] = values
    if not metrics:
        return None
    return steps, metrics


def merge_records(
    pieces: list[tuple[np.ndarray, dict[str, np.ndarray]]]
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Merge same-name runs (crash + restart) into one step -> metrics table.

    The piece with the most data wins on overlapping steps (largest last step,
    then most points); the others fill in whatever it does not cover, so a run
    covering [0, 500] and its restart covering [500, 1000] patch into [0, 1000].
    """
    order = sorted(
        range(len(pieces)),
        key=lambda i: (-float(pieces[i][0].max()), -len(pieces[i][0])),
    )
    merged: dict[float, dict[str, float]] = {}
    seen: set[float] = set()
    for i in order:
        steps, metrics = pieces[i]
        for row, step in enumerate(steps):
            key = float(step)
            slot = merged.setdefault(key, {})
            primary = key not in seen
            for name, values in metrics.items():
                value = float(values[row])
                if not np.isfinite(value):
                    continue
                if primary or name not in slot:
                    slot[name] = value
        seen.update(float(step) for step in steps)

    steps_out = np.array(sorted(merged), dtype=float)
    names = sorted({name for slot in merged.values() for name in slot})
    table: dict[str, np.ndarray] = {}
    for name in names:
        column = np.array(
            [merged[step].get(name, np.nan) for step in steps_out], dtype=float
        )
        table[name] = fill(column)
    return steps_out, table


def fill(values: np.ndarray) -> np.ndarray:
    """Forward-fill a metric, back-filling a leading gap (no empty CSV cells)."""
    if not np.isfinite(values).any():
        return np.zeros_like(values)
    index = np.where(np.isfinite(values), np.arange(len(values)), 0)
    np.maximum.accumulate(index, out=index)
    filled = values[index]
    first = int(np.argmax(np.isfinite(values)))
    if first:
        filled[:first] = filled[first]
    return filled


# --------------------------------------------------------------------------- #
# csv
# --------------------------------------------------------------------------- #
def sanitize(name: str, used: set[str]) -> str:
    """A LaTeX/CSV-safe column name, unique within the table."""
    base = re.sub(r"[^A-Za-z0-9]+", ".", name).strip(".")
    base = re.sub(r"\.{2,}", ".", base) or "metric"
    candidate, n = base, 2
    while candidate in used:
        candidate = f"{base}.{n}"
        n += 1
    used.add(candidate)
    return candidate


def write_run_csv(path: Path, steps: np.ndarray, table: dict[str, np.ndarray]) -> dict:
    """Write one run, all metrics as columns. Returns column metadata."""
    used: set[str] = set()
    columns = {"update": "update", "env": "env"}
    used.update(columns.values())
    for name in table:
        columns[name] = sanitize(name, used)

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["update", "env"] + [columns[name] for name in table])
        for row, step in enumerate(steps):
            values = [
                f"{float(step):.10g}",
                f"{float(step) * _ENV_FACTOR[0]:.10g}",
            ]
            values += [f"{float(table[name][row]):.8g}" for name in table]
            writer.writerow(values)

    return {
        "file": path.name,
        "rows": int(len(steps)),
        "update_start": float(steps.min()),
        "update_end": float(steps.max()),
        "columns": columns,
    }


_ENV_FACTOR = [DEFAULT_CAPACITY]  # capacity used for the env column


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--group",
        nargs="+",
        required=True,
        help="wandb group name(s) or shell-style pattern(s), e.g. 'final-a*-gt-virt'.",
    )
    parser.add_argument(
        "--out",
        default=str(output_root() / "csv"),
        help="Output directory (default: <repo or $HECA_OUT>/csv).",
    )
    parser.add_argument("--samples", type=int, default=10000, help="Max samples/run.")
    parser.add_argument(
        "--capacity",
        type=int,
        default=DEFAULT_CAPACITY,
        help=f"Option steps per update for the env column (default {DEFAULT_CAPACITY}).",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=None,
        help="Only these metric keys (default: every numeric metric of the run).",
    )
    parser.add_argument(
        "--tag",
        nargs="+",
        default=None,
        help="Group name(s) or patterns to record instead of the wandb group name, "
        "e.g. 'a0=final-a0-gt-virt' names the directory 'a0'.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the groups and runs that would be written, write nothing.",
    )
    args = parser.parse_args()

    _ENV_FACTOR[0] = args.capacity

    config = WandBConfig()
    api = wandb.Api()
    try:
        runs = list(
            api.runs(
                f"{config.entity}/{config.project}",
                filters={"group": {"$regex": _regex_any(args.group)}},
                per_page=200,
            )
        )
    except Exception:
        runs = list(api.runs(f"{config.entity}/{config.project}", per_page=200))

    groups = match_groups(runs, args.group)
    if not groups:
        available = sorted({g for run in runs if (g := group_of(run))})
        raise SystemExit(
            f"No runs matched {args.group} in {config.entity}/{config.project}.\n"
            f"Groups seen: {available[:20] or 'none'}"
        )

    renames: dict[str, str] = {}
    for entry in args.tag or []:
        tag, _, pattern = entry.partition("=")
        if not pattern:
            continue
        for group in groups:
            if fnmatch.fnmatch(group, pattern):
                renames[group] = tag

    out_root = Path(args.out)
    index: dict[str, dict] = {"entity": config.entity, "project": config.project,
                              "capacity": args.capacity, "groups": {}}
    group_lines: list[str] = []
    written = 0

    print(f"run csvs ({len(groups)} group(s)):")
    for group, members in groups.items():
        directory = out_root / renames.get(group, group)
        by_name: dict[str, list] = {}
        for run in members:
            by_name.setdefault(str(run.name), []).append(run)

        entries = []
        for name in sorted(by_name):
            name_runs = by_name[name]
            capacity = capacity_of(name_runs[0], args.capacity)
            pieces = []
            for run in name_runs:
                loaded = fetch_run(run, args.samples)
                if loaded is not None:
                    pieces.append(loaded)
            if not pieces:
                print(f"  {group:<26} {name:<34} no metrics, skipped")
                continue
            steps, table = merge_records(pieces)
            if args.metrics:
                table = select_metrics(table, args.metrics)
                if not table:
                    print(f"  {group:<26} {name:<34} none of {args.metrics}, skipped")
                    continue
            _ENV_FACTOR[0] = capacity

            print(
                f"  {group:<26} {name:<34} rows={len(steps):5d} "
                f"updates=[{steps.min():.0f},{steps.max():.0f}] "
                f"metrics={len(table)}"
            )
            if args.dry_run:
                continue

            info = write_run_csv(directory / f"{name}.csv", steps, table)
            info |= {
                "run_ids": sorted(str(r.id) for r in name_runs),
                "run_names": sorted(str(r.name) for r in name_runs),
                "states": sorted({str(getattr(r, "state", "unknown")) for r in name_runs}),
                "capacity": capacity,
            }
            entries.append(info)
            written += 1

        if args.dry_run or not entries:
            continue

        # longest run first: the LaTeX macro takes x from the first file
        entries.sort(key=lambda e: (-e["update_end"], e["file"]))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "index.json").write_text(
            json.dumps({"group": group, "runs": entries}, indent=2) + "\n"
        )
        index["groups"][group] = {
            "directory": str(directory),
            "files": [e["file"] for e in entries],
            "runs": [e["run_names"] for e in entries],
        }
        # paths relative to --out, so the file works when <out> is copied next
        # to the LaTeX document
        group_lines.append(
            f"{group}="
            + ",".join(str(Path(directory, e["file"]).relative_to(out_root)) for e in entries)
        )
        # flush after every group: an interrupted download still leaves a
        # consistent index and group list for everything written so far
        write_index(out_root, index, group_lines)

    if args.dry_run:
        print(f"\n(dry run: {sum(len(m) for m in groups.values())} run(s), nothing written)")
        return

    write_index(out_root, index, group_lines)

    print(
        f"\nSaved {written} run csv(s) under {out_root}\n"
        f"  {out_root / 'index.json'}\n"
        f"  {out_root / 'trainingsplot_groups.txt'}"
    )


def select_metrics(table: dict, wanted: list[str]) -> dict:
    """Metrics of ``table`` matching one of ``wanted`` (exact or by suffix)."""
    keep = {
        key
        for key in table
        if any(key == metric or key.endswith("/" + metric) for metric in wanted)
    }
    return {key: table[key] for key in table if key in keep}


def write_index(out_root: Path, index: dict, group_lines: list[str]) -> None:
    """Root index + the LaTeX group list (rewritten after every group)."""
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    (out_root / "trainingsplot_groups.txt").write_text("\n".join(group_lines) + "\n")


if __name__ == "__main__":
    main()
