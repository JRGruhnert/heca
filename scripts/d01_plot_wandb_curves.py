"""Plot training curves from wandb: mean and ±1 SD band across seeds.

Crash/restart handling: runs of the same name (a crashed run and its restart)
are merged into one series -- the run with the most data wins wherever their
steps overlap, and gaps are filled from the other runs, so a run covering
[0, 500] and its restart covering [500, 1000] patch into [0, 1000]. Runs whose
name belongs to the same wandb group (seeds ``_s0``/``_s1``/``_s2``) are then
aggregated as the mean with a ±1 SD band.

Plots are minimal on purpose: axis labels and a legend only when several groups
are drawn -- titles and descriptions are added in LaTeX.

    python -m scripts.d01_plot_wandb_curves --group 'final-a*-gt-virt' \
        --metric stats/success_rate --x update
"""

import argparse
import fnmatch
import json
import math
import re
import sys
from pathlib import Path

from heca.misc.paths import output_root

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np

from heca.misc.logger import WandBConfig

try:
    import wandb
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"wandb is not installed in the active environment: {exc}"
    ) from exc


DEFAULT_CAPACITY = 2048  # option steps per learner update (buffer capacity)


def group_of(run) -> str | None:
    """Group a run belongs to: its wandb group, else its name without the seed."""
    group = getattr(run, "group", None)
    if group:
        return str(group)
    name = str(run.name)
    stripped = re.sub(r"_s\d+$", "", name)
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


def load_run_series(
    run, metric: str, samples: int
) -> tuple[np.ndarray, np.ndarray] | None:
    """Update-step/value series of one metric, or None when the run has none."""
    base = re.sub(r"_s\d+$", "", str(run.name))
    candidates = [
        metric,
        f"{run.name}/{metric}",
        f"{metric}/{run.name}",
        f"{base}/{metric}",
        f"{metric}/{base}",
    ]
    for key in candidates:
        try:
            hist = run.history(keys=[key], samples=samples)
        except Exception:
            continue
        if hist is None or hist.empty or key not in hist.columns:
            continue
        col = hist[key]
        mask = col.notna().to_numpy()
        steps = hist["_step"].to_numpy(dtype=float)[mask]
        values = col.to_numpy(dtype=float)[mask]
        if len(steps):
            order = np.argsort(steps)
            return steps[order], values[order]
    return None


def capacity_of(run, fallback: int) -> int:
    """Buffer capacity logged with the run (option steps between two updates)."""
    config = getattr(run, "config", None) or {}
    for key in ("buffer/capacity", "buffer.capacity", "capacity"):
        value = config.get(key) if hasattr(config, "get") else None
        if isinstance(value, (int, float)) and value > 0:
            return int(value)
    return fallback


# --------------------------------------------------------------------------- #
# crash/restart handling and aggregation
# --------------------------------------------------------------------------- #
def merge_runs(series: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    """Merge duplicate runs of one name into a single series.

    The run with the most data (largest last step, then most points) wins
    wherever steps overlap; the others fill its gaps. A run covering [0, 500]
    and its restart covering [500, 1000] therefore patch into [0, 1000], and a
    falsely started run at [0, 5] next to a full [0, 1000] run contributes
    nothing.
    """
    if len(series) == 1:
        return series[0]
    order = sorted(
        range(len(series)),
        key=lambda i: (-float(series[i][0].max()), -len(series[i][0])),
    )
    merged: dict[float, float] = {}
    for i in order:
        for x, y in zip(series[i][0], series[i][1]):
            merged.setdefault(float(x), float(y))
    xs = np.array(sorted(merged), dtype=float)
    ys = np.array([merged[x] for x in xs], dtype=float)
    return xs, ys


def smooth(values: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average over ``window`` samples (edges use fewer)."""
    if window <= 1 or len(values) < 2:
        return values
    kernel = np.ones(min(window, len(values)))
    totals = np.convolve(values, kernel, mode="same")
    counts = np.convolve(np.ones(len(values)), kernel, mode="same")
    return totals / counts


def t_critical(n: int) -> float:
    """97.5% quantile of Student's t with ``n-1`` dof (95% CI of the mean)."""
    if n <= 1:
        return 1.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96  # normal approximation fallback


def common_grid(
    series: list[tuple[np.ndarray, np.ndarray]], n_points: int, min_seeds: int
) -> np.ndarray:
    """Grid up to the longest step covered by at least ``min_seeds`` runs."""
    ends = sorted((float(s.max()) for s, _ in series), reverse=True)
    needed = max(1, min(min_seeds or len(series), len(ends)))
    xmax = ends[needed - 1]
    return np.linspace(0.0, xmax, n_points)


def aggregate(
    series: list[tuple[np.ndarray, np.ndarray]], grid: np.ndarray, band: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean and half-width across runs on ``grid`` (band: sd | ci | none)."""
    stacked = np.asarray([np.interp(grid, s, v) for s, v in series])
    mean = stacked.mean(axis=0)
    if band == "none" or stacked.shape[0] < 2:
        return mean, np.zeros_like(mean), stacked
    sd = stacked.std(axis=0, ddof=1)
    half = (
        sd
        if band == "sd"
        else t_critical(stacked.shape[0]) * sd / math.sqrt(stacked.shape[0])
    )
    return mean, half, stacked


def summary(values: np.ndarray, tail_fraction: float) -> tuple[float, float]:
    """(mean of the last ``tail_fraction`` of a curve, value at its end)."""
    tail = max(1, int(round(len(values) * tail_fraction)))
    return float(values[-tail:].mean()), float(values[-1])


def short_labels(groups: list[str]) -> dict[str, str]:
    """Strip the prefix/suffix the groups share, for a compact legend."""
    if len(groups) < 2:
        return {g: g for g in groups}
    prefix = groups[0]
    for group in groups[1:]:
        while not group.startswith(prefix):
            prefix = prefix[:-1]
    suffix = groups[0]
    for group in groups[1:]:
        while suffix and not group.endswith(suffix):
            suffix = suffix[1:]  # trim the front: keep a common *suffix*
    labels = {}
    for group in groups:
        label = (
            group[len(prefix) : len(group) - len(suffix) or None]
            if suffix
            else group[len(prefix) :]
        )
        labels[group] = label.strip("-_") or group
    return labels


# --------------------------------------------------------------------------- #
# plotting
# --------------------------------------------------------------------------- #
def plot_metric(
    ax, metric: str, curves: dict[str, dict], x_label: str, seeds: bool, x_max=None
):
    colors = plt.get_cmap("tab10")
    for i, (group, curve) in enumerate(curves.items()):
        color = colors(i % 10)
        grid, mean, half = curve["grid"], curve["mean"], curve["half"]
        if seeds:
            for series in curve["stacked"]:
                ax.plot(grid, series, color=color, alpha=0.25, linewidth=1.0)
        ax.plot(grid, mean, color=color, label=curve["label"])
        if half.any():
            ax.fill_between(grid, mean - half, mean + half, alpha=0.2, color=color)

    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.removeprefix("stats/").removeprefix("train/"))
    ax.grid(True, alpha=0.3)
    if len(curves) > 1:
        ax.legend(fontsize=8)
    ends = [c["grid"][-1] for c in curves.values() if len(c["grid"])]
    if ends:
        # no margin: the axis runs from 0 to the end of the data exactly
        ax.set_xlim(0, float(x_max) if x_max is not None else max(ends))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        nargs="+",
        required=True,
        help="wandb group name(s) or shell-style pattern(s) of them, e.g. "
        "'final-a*-gt-virt'. Each matched group becomes one mean curve. "
        "An entry may carry its own legend tag as 'tag=pattern', e.g. "
        "'a0=final-a0-gt-virt' draws that group as 'a0'.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["stats/success_rate"],
        help="Metric keys to plot (default: stats/success_rate).",
    )
    parser.add_argument(
        "--out",
        default=str(output_root() / "plots"),
        help="Output directory (defaults to <repo or $HECA_OUT>/plots).",
    )
    parser.add_argument("--samples", type=int, default=10000, help="Max samples/run.")
    parser.add_argument(
        "--x",
        choices=["env", "update"],
        default="env",
        help="X axis: environment steps (updates x buffer capacity) or updates.",
    )
    parser.add_argument(
        "--capacity",
        type=int,
        default=DEFAULT_CAPACITY,
        help=f"Option steps per update when the run config does not say "
        f"(default {DEFAULT_CAPACITY}).",
    )
    parser.add_argument(
        "--x-max",
        type=float,
        default=None,
        help="Axis maximum (default: the end of the data, with no margin).",
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=0,
        help="Centered moving-average window in x units before aggregating "
        "(0 = raw). Use ~20 updates for training curves.",
    )
    parser.add_argument(
        "--band",
        choices=["sd", "ci", "none"],
        default="sd",
        help="Band across seeds: sd (±1 SD, default), ci (95%% CI of the mean) "
        "or none.",
    )
    parser.add_argument(
        "--seeds",
        action="store_true",
        help="Also draw each seed run as a faint line.",
    )
    parser.add_argument(
        "--min-seeds",
        type=int,
        default=0,
        help="Runs that must cover a point for it to be plotted (0 = all runs "
        "in the group; lower it to keep the full length of long runs).",
    )
    parser.add_argument(
        "--points", type=int, default=1000, help="Points on the common grid."
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Skip the PNG: write the raw JSON (mean/std arrays) only.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Also write a pgfplots-ready CSV per metric "
        "(columns: update, <group>_mean, <group>_std, <group>_lower, "
        "<group>_upper).",
    )
    parser.add_argument(
        "--tail",
        type=float,
        default=0.1,
        help="Fraction of the curve used for the 'last window' summary (0.1).",
    )
    parser.add_argument(
        "--short-labels",
        action="store_true",
        help="Strip the prefix/suffix shared by the groups in the legend.",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Ignore seed grouping: draw every run name as its own curve.",
    )
    args = parser.parse_args()

    entries = [
        (tag, pattern)
        for tag, pattern in (
            entry.split("=", 1) if "=" in entry else (None, entry)
            for entry in args.group
        )
    ]
    patterns = [pattern for _, pattern in entries]

    config = WandBConfig()
    api = wandb.Api()
    try:
        runs = list(
            api.runs(
                f"{config.entity}/{config.project}",
                filters={"group": {"$regex": _regex_any(patterns)}},
                per_page=200,
            )
        )
    except Exception:
        runs = list(api.runs(f"{config.entity}/{config.project}", per_page=200))

    groups = match_groups(runs, patterns)
    if not groups:
        available = sorted({g for run in runs if (g := group_of(run))})
        raise SystemExit(
            f"No runs matched {patterns} in {config.entity}/{config.project}.\n"
            f"Groups seen: {available[:20] or 'none'}"
        )

    # legend labels: an explicit tag wins, otherwise the group's own name
    label_map = {group: group for group in groups}
    for tag, pattern in ((t, p) for t, p in entries if t is not None):
        for group in groups:
            if fnmatch.fnmatch(group, pattern):
                label_map[group] = tag

    # per group: merge duplicate names (crashed + restarted), then aggregate seeds
    curves: dict[str, dict] = {}
    for metric in args.metric:
        curves[metric] = {}
    print(f"group aggregation ({len(groups)} group(s)):")
    for group, members in groups.items():
        # bucket the runs by name: duplicates of one name are one seed
        by_name: dict[str, list] = {}
        for run in members:
            by_name.setdefault(str(run.name), []).append(run)

        for metric in args.metric:
            series: list[tuple[np.ndarray, np.ndarray]] = []
            provenance: dict[str, list[str]] = {}
            for name, runs_of_name in by_name.items():
                pieces: list[tuple[np.ndarray, np.ndarray]] = []
                for run in runs_of_name:
                    loaded = load_run_series(run, metric, args.samples)
                    if loaded is None:
                        continue
                    steps, values = loaded
                    x = (
                        steps * capacity_of(run, args.capacity)
                        if args.x == "env"
                        else steps
                    )
                    pieces.append((x, values))
                if not pieces:
                    continue
                merged_x, merged_y = merge_runs(pieces)
                series.append((merged_x, merged_y))
                provenance[name] = sorted(
                    {str(r.name) for r in runs_of_name} - {name}
                ) or [name]

            if not series:
                continue
            if args.per_run:
                # one curve per run name instead of one curve per group
                for name, (sx, sy) in zip(provenance, series):
                    grid = np.linspace(sx.min(), sx.max(), args.points)
                    smoothed = (sx, smooth(sy, args.smooth))
                    mean, half, stacked = aggregate([smoothed], grid, args.band)
                    label = (
                        f"{label_map[group]}/{name}"
                        if len(groups) > 1 or len(series) > 1
                        else name
                    )
                    curves[metric][f"{group}/{name}"] = {
                        "grid": grid,
                        "mean": mean,
                        "half": half,
                        "stacked": stacked,
                        "runs": provenance[name],
                        "label": label,
                    }
                continue

            grid = common_grid(series, args.points, args.min_seeds)
            smoothed = [(s, smooth(v, args.smooth)) for s, v in series]
            mean, half, stacked = aggregate(smoothed, grid, args.band)
            curves[metric][group] = {
                "grid": grid,
                "mean": mean,
                "half": half,
                "stacked": stacked,
                "runs": list(provenance.keys()),
                "label": label_map[group],
            }
            print(
                f"  {group:<28} {metric:<24} seeds={len(series)} "
                f"(of {len(by_name)} names) x=[{grid[0]:.0f},{grid[-1]:.0f}]"
            )

    if not any(curves[m] for m in args.metric):
        raise SystemExit(f"No runs with metrics {args.metric} in {args.group}.")

    if args.short_labels:
        for metric in args.metric:
            labels = short_labels(list(curves[metric]))
            for group, curve in curves[metric].items():
                curve["label"] = (
                    label_map[group] if label_map[group] != group else labels[group]
                )

    x_label = "environment steps" if args.x == "env" else "updates"
    if not args.data_only:
        n_fig = len(args.metric)
        fig, axes = plt.subplots(n_fig, 1, figsize=(6, 3.4 * n_fig), squeeze=False)
        for ax, metric in zip(axes[:, 0], args.metric):
            if curves[metric]:
                plot_metric(ax, metric, curves[metric], x_label, args.seeds, args.x_max)
            else:
                ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)
        fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", "-".join(sorted(curves[args.metric[0]])))[:80]
    path = out_dir / f"curves_{stem}.png"
    if not args.data_only:
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)

    payload = {
        "groups": {},
        "x": args.x,
        "band": args.band,
        "smooth": args.smooth,
        "tail": args.tail,
        "metrics": {},
    }
    print(
        "\nsummary (mean over the last "
        f"{args.tail:.0%} of the curve, and at its end):"
    )
    for group, members in groups.items():
        names = sorted({str(run.name) for run in members})
        payload["groups"][group] = {
            "runs": names,
            "n_names": len(names),
            "n_runs": len(members),
        }
    for metric in args.metric:
        payload["metrics"][metric] = {}
        for group, curve in curves[metric].items():
            tail_mean, tail_end = summary(curve["mean"], args.tail)
            tail_sd = (
                float(
                    curve["stacked"][
                        :, -max(1, int(round(len(curve["mean"]) * args.tail))) :
                    ]
                    .mean(axis=1)
                    .std(ddof=1)
                )
                if curve["stacked"].shape[0] > 1
                else 0.0
            )
            payload["metrics"][metric][group] = {
                "label": curve["label"],
                "n_seeds": curve["stacked"].shape[0],
                "tail_mean": tail_mean,
                "tail_sd_across_seeds": tail_sd,
                "final": tail_end,
                "grid": curve["grid"].tolist(),
                "mean": curve["mean"].tolist(),
                **({"std": curve["half"].tolist()} if args.band == "sd" else {}),
                "band_half_width": curve["half"].tolist(),
                "runs": curve["runs"],
            }
            print(
                f"  {label_map[group]:<28} {metric:<24} "
                f"{100 * tail_mean if 'rate' in metric else tail_mean:8.4f}"
                f" ± {100 * tail_sd if 'rate' in metric else tail_sd:.4f}"
                f"  (n={curve['stacked'].shape[0]})"
            )
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    saved = [str(json_path)] if args.data_only else [str(path), str(json_path)]

    if args.csv:
        import csv as csv_module

        for metric in args.metric:
            metric_groups = list(curves[metric])
            if not metric_groups:
                continue
            grids = [curves[metric][g]["grid"] for g in metric_groups]
            shared = all(np.allclose(grids[0], g) for g in grids[1:])
            csv_path = out_dir / f"curves_{stem}_{metric.replace('/', '-')}.csv"
            with csv_path.open("w", newline="") as fh:
                writer = csv_module.writer(fh)
                std_of = (
                    (lambda c, i: c["half"][i]) if args.band == "sd" else (lambda c, i: 0.0)
                )
                if shared:
                    writer.writerow(
                        ["update"]
                        + [
                            f"{curves[metric][g]['label']}_{suffix}"
                            for g in metric_groups
                            for suffix in ("mean", "std", "lower", "upper")
                        ]
                    )
                    for i in range(len(grids[0])):
                        row = [f"{grids[0][i]:.6g}"]
                        for g in metric_groups:
                            c = curves[metric][g]
                            mean = c["mean"][i]
                            std = std_of(c, i)
                            row += [
                                f"{mean:.8g}",
                                f"{std:.8g}",
                                f"{mean - std:.8g}",
                                f"{mean + std:.8g}",
                            ]
                        writer.writerow(row)
                else:
                    writer.writerow(
                        [
                            f"{curves[metric][g]['label']}_{suffix}"
                            for g in metric_groups
                            for suffix in ("update", "mean", "std", "lower", "upper")
                        ]
                    )
                    length = max(len(c["grid"]) for c in [curves[metric][g] for g in metric_groups])
                    for i in range(length):
                        row = []
                        for g in metric_groups:
                            c = curves[metric][g]
                            if i < len(c["grid"]):
                                mean = c["mean"][i]
                                std = std_of(c, i)
                                row += [
                                    f"{c['grid'][i]:.6g}",
                                    f"{mean:.8g}",
                                    f"{std:.8g}",
                                    f"{mean - std:.8g}",
                                    f"{mean + std:.8g}",
                                ]
                            else:
                                row += ["", "", "", "", ""]
                        writer.writerow(row)
            saved.append(str(csv_path))

    print(f"\nSaved " + "\nSaved ".join(saved))


def _regex_any(patterns: list[str]) -> str:
    """Mongo-style regex matching any of the shell patterns (server-side filter)."""
    return "|".join(
        fnmatch.translate(p).removeprefix("(?s:").removesuffix(")\\Z") for p in patterns
    )


if __name__ == "__main__":
    main()
