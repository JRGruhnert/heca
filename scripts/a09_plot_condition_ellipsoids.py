import argparse
import sys
from pathlib import Path

# Make ``conf`` and ``scripts.common`` importable when run directly.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import numpy as np
from scipy.stats import chi2

from heca.experts.expert import ExpertModel
from heca.misc import logger

from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import agents_by_scene

# The three default position projections: xy, xz, yz.
_DEFAULT_DIMS = [(0, 1), (0, 2), (1, 2)]

# Quantile levels drawn as nested ellipses per component (innermost = filled).
_DEFAULT_QUANTILES = (0.95, 0.99, 0.999)
_QUANTILE_LINESTYLES = ("-", "--", ":")


def pose_dim_names(entity) -> list[str]:
    """Names of the model-space pose dimensions (state column excluded)."""
    names = ["pos_x", "pos_y", "pos_z"]
    if entity.cfg.fit_rotation:
        names += ["rot_x", "rot_y", "rot_z"]
    names += ["extra_0", "extra_1", "extra_2", "extra_3"]
    return names


def sample_env_values(entity, scene, con, label: str, n: int):
    """Sample ``n`` values for one entity from the environment.

    Calls the (assumed) ``scene.samples(n)`` and returns the model-space pose
    matrix ``(n, pose_dims)`` (state column dropped). Falls back to the
    condition's recorded values (``con.data_raw[label]``) if ``scene.samples``
    does not exist.
    """
    if hasattr(scene, "samples"):
        raw = scene.samples(n)
        arr = np.atleast_2d(np.asarray(raw[label], dtype=np.float64))
        # Values are in the full layout [pos, aa, extra, ste]; map to model
        # space (drops rotation columns when fit_rotation=False). If they are
        # already model-space sized, model_value is the identity anyway.
        arr = entity.model_value(arr)
        return arr[:, :-1]  # drop the state column
    logger.warning(
        f"scene.samples(n) not implemented on {type(scene).__name__}; "
        "falling back to the condition's recorded values (data_raw)."
    )
    return condition_data_values(entity, con, label, n)


def condition_data_values(entity, con, label: str, n: int):
    """Fallback: the condition's recorded demo values, model space, state
    column dropped (at most ``n`` rows)."""
    arr = np.asarray(con.data_raw[label], dtype=np.float64)
    if len(arr) > n:
        arr = arr[:n]
    arr = entity.model_value(arr)
    return arr[:, :-1]


def component_of(entity, pose: np.ndarray, p) -> int:
    """Best-matching component index for a pose (weight * Gaussian posterior)."""
    return entity._best_component(pose, p)[0]


def _ellipse(ax, mu, var, dims, r, color, filled: bool, linestyle: str):
    a, b = dims
    ax.add_patch(
        Ellipse(
            (mu[a], mu[b]),
            width=2 * np.sqrt(var[a]) * r,
            height=2 * np.sqrt(var[b]) * r,
            facecolor=color if filled else "none",
            edgecolor="k" if filled else color,
            alpha=0.15 if filled else 1.0,
            linestyle=linestyle,
            linewidth=1.0 if filled else 1.2,
        )
    )


def quantile_handles(quantiles):
    """Proxy legend handles for the quantile levels."""
    from matplotlib.patches import Patch

    return [
        Patch(
            facecolor="none",
            edgecolor="0.2",
            linestyle=_QUANTILE_LINESTYLES[i % len(_QUANTILE_LINESTYLES)],
            label=f"{q * 100:g}%",
        )
        for i, q in enumerate(sorted(quantiles))
    ]


def plot_projection(ax, entity, pose, p, dims, quantiles, title=""):
    """2D scatter of samples (colored by component) + nested component
    ellipses at the given quantile levels (innermost filled)."""
    a, b = dims
    n_comp = len(p["weights"])
    colors = plt.get_cmap("tab10")
    d = p["measurement"]["pose"]["means"].shape[1]
    names = pose_dim_names(entity)
    qs = sorted(quantiles)

    comps = np.array([component_of(entity, pose[i], p) for i in range(len(pose))])
    for k in range(n_comp):
        mu = p["measurement"]["pose"]["means"][k]
        var = p["measurement"]["pose"]["covariances"][k]
        for i, q in enumerate(qs):
            r = float(np.sqrt(chi2.ppf(q, d)))
            _ellipse(
                ax,
                mu,
                var,
                dims,
                r,
                colors(k % 10),
                filled=(i == 0),
                linestyle=_QUANTILE_LINESTYLES[i],
            )
        mask = comps == k
        if np.any(mask):
            ax.scatter(
                pose[mask, a],
                pose[mask, b],
                s=8,
                color=colors(k % 10),
                alpha=0.6,
                label=f"comp{k} (w={p['weights'][k]:.2f})",
            )

    ax.set_title(title)
    ax.set_xlabel(names[a] if a < len(names) else f"dim {a}")
    ax.set_ylabel(names[b] if b < len(names) else f"dim {b}")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, loc="upper right")


def plot_matrix_row(ax_row, entity, pose, p, quantiles):
    """Pairwise scatter matrix (one row of axes, one panel per pose dim)."""
    d = pose.shape[1]
    n_comp = len(p["weights"])
    colors = plt.get_cmap("tab10")
    names = pose_dim_names(entity)
    qs = sorted(quantiles)
    comps = np.array([component_of(entity, pose[i], p) for i in range(len(pose))])

    for i in range(d):
        for j in range(d):
            axij = ax_row[i] if d > 1 else ax_row[0]
            if i == j:
                axij.hist(pose[:, i], bins=25, color="0.7", alpha=0.6)
                for k in range(n_comp):
                    mu = p["measurement"]["pose"]["means"][k]
                    var = p["measurement"]["pose"]["covariances"][k]
                    axij.axvline(mu[i], color=colors(k % 10))
                    for q in qs:
                        r = float(np.sqrt(chi2.ppf(q, d)))
                        axij.axvline(
                            mu[i] - np.sqrt(var[i]) * r,
                            color=colors(k % 10),
                            linestyle="--",
                            linewidth=0.8,
                        )
                        axij.axvline(
                            mu[i] + np.sqrt(var[i]) * r,
                            color=colors(k % 10),
                            linestyle="--",
                            linewidth=0.8,
                        )
            else:
                for k in range(n_comp):
                    mu = p["measurement"]["pose"]["means"][k]
                    var = p["measurement"]["pose"]["covariances"][k]
                    for iq, q in enumerate(qs):
                        r = float(np.sqrt(chi2.ppf(q, d)))
                        _ellipse(
                            axij,
                            mu,
                            var,
                            (j, i),
                            r,
                            colors(k % 10),
                            filled=(iq == 0),
                            linestyle=_QUANTILE_LINESTYLES[iq],
                        )
                    mask = comps == k
                    if np.any(mask):
                        axij.scatter(
                            pose[mask, j],
                            pose[mask, i],
                            s=4,
                            color=colors(k % 10),
                            alpha=0.5,
                        )
            if j == 0:
                axij.set_ylabel(names[i] if i < len(names) else f"dim {i}")
            if i == d - 1:
                axij.set_xlabel(names[j] if j < len(names) else f"dim {j}")


def _value_z2(entity, pose, p, eps: float = 1e-15) -> np.ndarray:
    """Per-value squared Mahalanobis distance to its best-matching component."""
    means = p["measurement"]["pose"]["means"]
    vars_ = np.maximum(p["measurement"]["pose"]["covariances"], eps)
    comps = np.array(
        [entity._best_component(pose[i], p)[0] for i in range(len(pose))]
    )
    return np.array(
        [
            float(np.sum((pose[i] - means[comps[i]]) ** 2 / vars_[comps[i]]))
            for i in range(len(pose))
        ]
    )


def plot_membership(ax, entity, pose, p, quantiles, title=""):
    """Histogram of each value's z^2 (Mahalanobis^2 to its best component) with
    the chi-square quantile cutoffs, and the fraction of values inside each
    level — i.e. directly "is this value inside its ellipsoid?"."""
    d = pose.shape[1]
    z2 = _value_z2(entity, pose, p)
    cuts = {q: float(chi2.ppf(q, d)) for q in sorted(quantiles)}

    xmax = max(float(np.max(z2)) if len(z2) else 1.0, max(cuts.values()) * 1.2)
    z2_hist = np.maximum(z2, 1e-4)  # floor so exact matches land in the first bin
    bins = np.logspace(np.log10(1e-4), np.log10(max(xmax, 1.0)), 80)
    ax.hist(z2_hist, bins=bins, color="0.7", alpha=0.7)
    ax.set_xscale("log")

    for q, c in cuts.items():
        ax.axvline(c, color="C3", linestyle="--", linewidth=1.2)
        ax.text(c, ax.get_ylim()[1] * 0.95, f" {q * 100:g}%",
                color="C3", fontsize=8, rotation=90, va="top")

    n = len(z2)
    lines = [f"n={n}"]
    prev = 0.0
    for q, c in cuts.items():
        inside = int(np.sum(z2 <= c))
        lines.append(f"≤{q * 100:g}%: {inside}/{n} ({100.0 * inside / n:.0f}%)")
        prev = c
    outside = int(np.sum(z2 > prev))
    lines.append(f"outside: {outside}/{n} ({100.0 * outside / n:.0f}%)")

    ax.set_xlabel("z² (Mahalanobis² to best component)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, which="both")
    ax.text(0.98, 0.98, "\n".join(lines), transform=ax.transAxes, ha="right",
            va="top", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.85))


def agent_plot(agent, scene, pair, scene_tag, n, dims, quantiles, out_dir,
               use_matrix, use_membership):
    """Plot one agent's pre/post conditions per shared entity."""
    for label in sorted(set(pair.pre.entities) & set(pair.post.entities)):
        entity = pair.pre.entities[label]
        q_lab = "/".join(f"{q * 100:g}%" for q in sorted(quantiles))

        pre_pose = sample_env_values(entity, scene, pair.pre, label, n)
        post_pose = sample_env_values(entity, scene, pair.post, label, n)
        pre_p = entity.secure_mix_parameters(pair.pre.models[label].get_parameters())
        post_p = entity.secure_mix_parameters(pair.post.models[label].get_parameters())

        base = f"{scene_tag} / {agent.cfg.tag} / {label}"
        if use_membership:
            fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
            plot_membership(axes[0], entity, pre_pose, pre_p, quantiles,
                            title=f"{base} — pre")
            plot_membership(axes[1], entity, post_pose, post_p, quantiles,
                            title=f"{base} — post")
            fig.suptitle(f"{base} — membership (values inside their ellipsoids)")
        elif use_matrix:
            d = pre_pose.shape[1]
            fig, axes = plt.subplots(2, d, figsize=(3.0 * d, 6.0), squeeze=False)
            plot_matrix_row(axes[0], entity, pre_pose, pre_p, quantiles)
            plot_matrix_row(axes[1], entity, post_pose, post_p, quantiles)
            fig.suptitle(f"{base} — pre (top) / post (bottom)")
        elif dims is None:
            # Default: the three position projections xy, xz, yz.
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            for col, (a, b) in enumerate(_DEFAULT_DIMS):
                plot_projection(
                    axes[0, col],
                    entity,
                    pre_pose,
                    pre_p,
                    (a, b),
                    quantiles,
                    title=f"{base} — pre "
                    f"({_dim_name(entity, a)}-{_dim_name(entity, b)})",
                )
                plot_projection(
                    axes[1, col],
                    entity,
                    post_pose,
                    post_p,
                    (a, b),
                    quantiles,
                    title=f"{base} — post "
                    f"({_dim_name(entity, a)}-{_dim_name(entity, b)})",
                )
            fig.suptitle(f"{base}")
        else:
            a, b = dims
            fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
            plot_projection(
                axes[0],
                entity,
                pre_pose,
                pre_p,
                (a, b),
                quantiles,
                title=f"{base} — pre ({_dim_name(entity, a)}-"
                f"{_dim_name(entity, b)})",
            )
            plot_projection(
                axes[1],
                entity,
                post_pose,
                post_p,
                (a, b),
                quantiles,
                title=f"{base} — post ({_dim_name(entity, a)}-"
                f"{_dim_name(entity, b)})",
            )
            fig.suptitle(f"{base} — dims {a},{b}")

        # Shared quantile legend at the bottom of the figure (not needed for
        # the membership view, which annotates the levels directly).
        if not use_membership:
            fig.legend(
                handles=quantile_handles(quantiles),
                loc="lower center",
                ncol=len(quantiles),
                fontsize=9,
                title=f"ellipsoid quantiles ({q_lab})",
            )
        fig.tight_layout(rect=(0, 0.06, 1, 0.98))

        path = out_dir / f"cond.png"
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {path}")


def _dim_name(entity, idx: int) -> str:
    names = pose_dim_names(entity)
    return names[idx] if idx < len(names) else f"dim{idx}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Refit conditions from demos instead of cache.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=300,
        help="Samples drawn per entity (scene.samples(n)).",
    )
    parser.add_argument(
        "--dims",
        type=int,
        nargs=2,
        default=None,
        help="Override the default xy/xz/yz projections with a "
        "single projection onto pose dims a b.",
    )
    parser.add_argument(
        "--quantiles",
        type=float,
        nargs="+",
        default=None,
        help="Ellipsoid quantile levels per component " "(default: 0.95 0.99 0.999).",
    )
    parser.add_argument(
        "--matrix",
        action="store_true",
        help="Show the full pairwise scatter matrix instead of "
        "the position projections.",
    )
    parser.add_argument(
        "--membership",
        action="store_true",
        help="Instead of projections, show for each value its Mahalanobis^2 "
        "to its best component with the chi-square quantile cutoffs — i.e. "
        "directly whether the values are inside their ellipsoids (and which "
        "quantile level).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory override. Default: the model's "
        "own directory (ExpertModel.save_dir(cfg)/plots), like "
        "the other plot scripts.",
    )
    args = parser.parse_args()

    quantiles = tuple(args.quantiles) if args.quantiles else _DEFAULT_QUANTILES
    matched = False
    for scene_cfg, scene_models in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        matched = True
        model_cfgs = scene_models
        if args.model:
            model_cfgs = [c for c in scene_models if c.tag == args.model]
            if not model_cfgs:
                logger.warning(
                    f"No agent {args.model!r} in scene {scene_cfg.tag}"
                )
                continue
        for cfg in model_cfgs:
            try:
                agent = ExpertModel.get(cfg, auto_load=False)
                if not args.gt:
                    agent.use_gt(False)
                if args.reload:
                    agent.force_recompute()
                agent.load()
                pair = agent.conditions
                out_dir = (
                    Path(args.out)
                    if args.out
                    else ExpertModel.save_dir(cfg) / "plots"
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                agent_plot(
                    agent,
                    agent.scene,
                    pair,
                    scene_cfg.tag,
                    args.n_samples,
                    args.dims,
                    quantiles,
                    out_dir,
                    args.matrix,
                    args.membership,
                )
            except Exception as exc:
                logger.error(f"[{scene_cfg.tag}] agent {cfg.tag} failed: {exc}")

    if not matched:
        available = [sc.tag for sc, _ in agents_by_scene()]
        hint = "none — are the scene modules commented out?"
        parser.error(
            f"Scene {args.scene!r} not found in conf/scenes.SCENE_MODULES. "
            f"Available scenes: {available or hint}"
        )


if __name__ == "__main__":
    main()
