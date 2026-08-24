import argparse
import math
import re
import sys
from pathlib import Path

# Make ``conf`` / ``scripts.common`` importable when run directly as
# ``python scripts/plot_wandb_curves.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np

from heca.learning.learner import WandBConfig

try:
    import wandb
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"wandb is not installed in the active environment: {exc}"
    ) from exc


def scene_of(run_name: str) -> str | None:
    """Scene client tag from a run name like ``exp_scene1`` -> ``scene1``."""
    m = re.search(r"_([^_]+)$", run_name)
    return m.group(1) if m else None


def load_run_series(run, metric: str, samples: int):
    base = re.sub(r"_s\d+$", "", run.name)
    candidates = {
        f"{run.name}/{metric}",
        f"{metric}/{run.name}",
        f"{base}/{metric}",
        f"{metric}/{base}",
    }
    for key in candidates:
        try:
            hist = run.history(keys=[key], samples=samples)
        except Exception:
            continue
        if hist is None or hist.empty or key not in hist.columns:
            continue
        col = hist[key]
        mask = col.notna()
        steps = hist["_step"].to_numpy(dtype=float)[mask.to_numpy()]
        values = col.to_numpy(dtype=float)[mask.to_numpy()]
        if len(steps) == 0:
            continue
        return steps, values
    return None


def t_critical(n: int) -> float:
    """97.5% quantile of Student's t with ``n-1`` dof (95% CI of the mean)."""
    if n <= 1:
        return 1.0
    try:
        from scipy.stats import t

        return float(t.ppf(0.975, df=n - 1))
    except Exception:
        return 1.96  # normal approximation fallback


def mean_ci(series_list: list[tuple[np.ndarray, np.ndarray]], n_points: int = 400):
    max_step = max(float(s.max()) for s, _ in series_list)
    grid = np.linspace(0.0, max_step, n_points)
    interp = [np.interp(grid, s, v) for s, v in series_list]
    arr = np.asarray(interp)
    mean = arr.mean(axis=0)
    if arr.shape[0] > 1:
        std = arr.std(axis=0, ddof=1)
        ci = t_critical(arr.shape[0]) * std / math.sqrt(arr.shape[0])
    else:
        ci = np.zeros_like(mean)
    return grid, mean, ci


def plot_metric(
    ax,
    metric: str,
    by_scene: dict[str, list[tuple[str, np.ndarray, np.ndarray]]],
    per_run: bool = False,
):
    """Plot one metric for all scene clients.

    Default: one mean+CI line per scene (aggregated across its runs).
    With ``per_run=True``: every run is its own line, colored per scene.
    """
    scenes = sorted(by_scene)
    colors = plt.get_cmap("tab10")

    if per_run:
        for i, scene in enumerate(scenes):
            color = colors(i % 10)
            for name, steps, values in by_scene[scene]:
                ax.plot(steps, values, color=color, alpha=0.7, linewidth=1.0)
            ax.plot([], [], color=color, linewidth=2.0, label=scene)
    else:
        for i, scene in enumerate(scenes):
            color = colors(i % 10)
            series = [(s, v) for _, s, v in by_scene[scene]]
            grid, mean, ci = mean_ci(series)
            ax.plot(grid, mean, color=color, label=scene)
            ax.fill_between(grid, mean - ci, mean + ci, alpha=0.25, color=color)

    ax.set_xlabel("run step")
    ax.set_ylabel(metric.removeprefix("stats/").removeprefix("train/"))
    ax.set_title(metric + (" (per run)" if per_run else " (mean ± 95% CI)"))
    ax.grid(True, alpha=0.3)
    if len(scenes) <= 10:
        ax.legend(fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--entity",
        default=WandBConfig().entity,
        help=f"wandb entity (default: {WandBConfig().entity}).",
    )
    parser.add_argument(
        "--project",
        default=WandBConfig().project,
        help=f"wandb project (default: {WandBConfig().project}).",
    )
    parser.add_argument(
        "--group",
        required=True,
        help="wandb group of the runs to aggregate (the run tag prefix, e.g. exp).",
    )
    parser.add_argument(
        "--metric",
        nargs="+",
        default=["stats/success_rate"],
        help="Metric keys to plot (default: stats/success_rate).",
    )
    parser.add_argument(
        "--out",
        default="plots",
        help="Output directory for the plot (default: ./plots).",
    )
    parser.add_argument(
        "--per-run",
        action="store_true",
        help="Plot every run as its own line instead of aggregating runs per "
        "client into a mean + CI line.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10000,
        help="Max history samples per run fetched from wandb.",
    )
    args = parser.parse_args()

    api = wandb.Api()
    runs = api.runs(f"{args.entity}/{args.project}", filters={"group": args.group})
    if not runs:
        raise SystemExit(
            f"No runs found for group={args.group!r} in "
            f"{args.entity}/{args.project}."
        )

    # scene -> metric -> list of (run name, steps, values)
    by_scene: dict[str, dict[str, list[tuple[str, np.ndarray, np.ndarray]]]] = {}
    for run in runs:
        scene = scene_of(run.name)
        if scene is None:
            continue
        for metric in args.metric:
            series = load_run_series(run, metric, args.samples)
            if series is not None:
                by_scene.setdefault(scene, {}).setdefault(metric, []).append(
                    (run.name, series[0], series[1])
                )
                print(
                    f"  {run.name:<24} scene={scene} "
                    f"steps={len(series[0])} metric={metric}"
                )

    if not by_scene:
        raise SystemExit(
            f"No runs with metric {args.metric} found for group={args.group!r}."
        )

    n_fig = len(args.metric)
    fig, axes = plt.subplots(n_fig, 1, figsize=(9, 4.2 * n_fig), squeeze=False)
    for ax, metric in zip(axes[:, 0], args.metric):
        if metric in {m for c in by_scene.values() for m in c}:
            plot_metric(
                ax,
                metric,
                {c: by_scene[c][metric] for c in by_scene if metric in by_scene[c]},
                per_run=args.per_run,
            )
        else:
            ax.set_title(f"{metric} (no data)")
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes
            )

    fig.suptitle(
        f"Training curves — group {args.group} " f"(mean ± 95% CI across runs)",
        fontsize=14,
    )
    fig.tight_layout()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"curves_{args.group}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
