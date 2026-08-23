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


def client_id(run_name: str) -> int | None:
    """Client index from a run name like ``exp_heca3_s1`` -> 3."""
    m = re.search(r"_heca(\d+)", run_name)
    return int(m.group(1)) if m else None


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
    by_client: dict[int, list[tuple[np.ndarray, np.ndarray]]],
):
    """Plot one metric: a mean+CI line per client."""
    for cid in sorted(by_client):
        grid, mean, ci = mean_ci(by_client[cid])
        (line,) = ax.plot(grid, mean, label=f"heca{cid}")
        ax.fill_between(grid, mean - ci, mean + ci, alpha=0.25, color=line.get_color())
    ax.set_xlabel("run step")
    ax.set_ylabel(metric.removeprefix("stats/").removeprefix("train/"))
    ax.set_title(metric)
    ax.grid(True, alpha=0.3)
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
        help="wandb group of the runs to aggregate (the part before _hecaN).",
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

    # client id -> metric -> list of (steps, values) per run
    by_client: dict[int, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    for run in runs:
        cid = client_id(run.name)
        if cid is None:
            continue
        for metric in args.metric:
            series = load_run_series(run, metric, args.samples)
            if series is not None:
                by_client.setdefault(cid, {}).setdefault(metric, []).append(series)
                print(
                    f"  {run.name:<24} client=heca{cid} "
                    f"steps={len(series[0])} metric={metric}"
                )

    if not by_client:
        raise SystemExit(
            f"No runs with metric {args.metric} found for group={args.group!r}."
        )

    n_fig = len(args.metric)
    fig, axes = plt.subplots(n_fig, 1, figsize=(9, 4.2 * n_fig), squeeze=False)
    for ax, metric in zip(axes[:, 0], args.metric):
        if metric in {m for c in by_client.values() for m in c}:
            plot_metric(
                ax,
                metric,
                {c: by_client[c][metric] for c in by_client if metric in by_client[c]},
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
