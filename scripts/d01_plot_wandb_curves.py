import argparse
import fnmatch
import json
import math
import re
import sys
from pathlib import Path

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
# aggregation
# --------------------------------------------------------------------------- #
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
    ax, metric: str, curves: dict[str, dict], x_label: str, band: str, seeds: bool
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

    band_label = {"sd": "±1 SD", "ci": "95% CI", "none": "no band"}[band]
    ax.set_xlabel(x_label)
    ax.set_ylabel(metric.removeprefix("stats/").removeprefix("train/"))
    ax.set_title(f"{metric} — mean across seeds ({band_label})")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, title="group", title_fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--group",
        nargs="+",
        required=True,
        help="wandb group name(s) or shell-style pattern(s) of them, e.g. "
        "'final-a*-gt-virt'. Each matched group becomes one mean curve.",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["stats/success_rate"],
        help="Metric keys to plot (default: stats/success_rate).",
    )
    parser.add_argument("--out", default="plots", help="Output directory (./plots).")
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
        "--points", type=int, default=400, help="Points on the common grid."
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
        help="Ignore grouping: draw every run as its own curve.",
    )
    args = parser.parse_args()

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
    if args.per_run:
        groups = {group: members for group, members in groups.items()}
    if not groups:
        available = sorted({g for run in runs if (g := group_of(run))})
        raise SystemExit(
            f"No runs matched {args.group} in {config.entity}/{config.project}.\n"
            f"Groups seen: {available[:20] or 'none'}"
        )

    curves: dict[str, dict] = {}
    for metric in args.metric:
        curves[metric] = {}
    print(f"group aggregation ({len(groups)} group(s)):")
    for group, members in groups.items():
        for metric in args.metric:
            series, names, grid_list = [], [], []
            for run in members:
                loaded = load_run_series(run, metric, args.samples)
                if loaded is None:
                    continue
                steps, values = loaded
                x = (
                    steps * capacity_of(run, args.capacity)
                    if args.x == "env"
                    else steps
                )
                series.append((x, values))
                names.append(str(run.name))
            if not series:
                continue
            grid = common_grid(series, args.points, args.min_seeds)
            smoothed = [(s, smooth(v, args.smooth)) for s, v in series]
            mean, half, stacked = aggregate(smoothed, grid, args.band)
            curves[metric][group] = {
                "grid": grid,
                "mean": mean,
                "half": half,
                "stacked": stacked,
                "runs": names,
                "label": group,
            }
            print(
                f"  {group:<28} {metric:<24} seeds={len(series)} "
                f"(of {len(members)}) x=[{grid[0]:.0f},{grid[-1]:.0f}]"
            )

    if not any(curves[m] for m in args.metric):
        raise SystemExit(f"No runs with metrics {args.metric} in {args.group}.")

    if args.short_labels:
        for metric in args.metric:
            labels = short_labels(list(curves[metric]))
            for group, curve in curves[metric].items():
                curve["label"] = labels[group]

    x_label = "environment steps" if args.x == "env" else "updates"
    n_fig = len(args.metric)
    fig, axes = plt.subplots(n_fig, 1, figsize=(9, 4.4 * n_fig), squeeze=False)
    for ax, metric in zip(axes[:, 0], args.metric):
        if curves[metric]:
            plot_metric(ax, metric, curves[metric], x_label, args.band, args.seeds)
        else:
            ax.set_title(f"{metric} (no data)")
            ax.text(0.5, 0.5, "no data", ha="center", va="transAxes")
    band_label = {"sd": "±1 SD", "ci": "95% CI", "none": "no band"}[args.band]
    seeds_info = ", ".join(
        f"{g}: n={len(c['runs'])}" for g, c in next(iter(curves.values())).items()
    )
    note = (
        f"{seeds_info} | seeds=mean across runs, band={band_label} | "
        f"smooth={'moving avg ' + str(args.smooth) + ' ' + ('updates' if args.x == 'update' else 'points') if args.smooth > 1 else 'off'} | "
        f"x={x_label}"
    )
    if any(m.startswith("stats/") for m in args.metric):
        note += (
            "\n'stats/*' comes from training rollouts (buffer statistics), "
            "not from an evaluation protocol."
        )
    fig.suptitle("Training curves — group aggregation", fontsize=14)
    fig.text(0.5, 0.005, note, ha="center", fontsize=8, color="0.25")
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", "-".join(sorted(groups)))[:80]
    path = out_dir / f"curves_{stem}.png"
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
        payload["groups"][group] = {
            "runs": [str(run.name) for run in members],
            "n_seeds": len(members),
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
                "n_seeds": curve["stacked"].shape[0],
                "tail_mean": tail_mean,
                "tail_sd_across_seeds": tail_sd,
                "final": tail_end,
                "grid": curve["grid"].tolist(),
                "mean": curve["mean"].tolist(),
                "band_half_width": curve["half"].tolist(),
                "runs": curve["runs"],
            }
            print(
                f"  {group:<28} {metric:<24} "
                f"{100 * tail_mean if 'rate' in metric else tail_mean:8.4f}"
                f" ± {100 * tail_sd if 'rate' in metric else tail_sd:.4f}"
                f"  (n={curve['stacked'].shape[0]})"
            )
    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"\nSaved {path}\nSaved {json_path}")


def _regex_any(patterns: list[str]) -> str:
    """Mongo-style regex matching any of the shell patterns (server-side filter)."""
    return "|".join(
        fnmatch.translate(p).removeprefix("(?s:").removesuffix(")\\Z") for p in patterns
    )


if __name__ == "__main__":
    main()
