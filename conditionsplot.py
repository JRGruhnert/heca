#!/usr/bin/env python3
"""Views of a model's conditions, one cloud per segment component.

The conditions of an expert are a mixture per entity: one component per segment
the demos were cut into, so plotting the components (instead of thousands of
demo values) keeps the picture readable. Two views are available:

``3d`` (the original): in metric space, the mean of every component as a point,
the 1-sigma covariance as a wireframe ellipsoid, and an optional light scatter of
the demo values assigned to the component. Axes are fixed to ``--lim`` metres.

``2d`` (TAPAS style, per axis): like the TAPAS prediction plot, but with the
component index where TAPAS has time -- one panel per dimension, the component
mean with its +-1-sigma error bar per component. Position gets the x, y and z
panels, the extra dims (prismatic slide / revolute angle) their own panel, and
the discrete state its probability panel.

    python conditionsplot.py --scene scene0 --model cube0_base_base --style 2d
    python conditionsplot.py --scene scene0 --model drawer0_a_b --style both
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Ellipse

from heca.experts.expert import ExpertModel
from heca.misc import logger
from heca.scenes.scene import Scene

from scripts.common.args import add_model_argument, add_scene_argument, add_use_gt_argument
from scripts.common.scenes import experts_for_scene, scene_config


# ============================================================================
#  3d view (unchanged)
# ============================================================================
def shells(ax, mean, var, colour, radius, n_std=1.0, post=False):
    """Soft Gaussian cloud: filled Mahalanobis shells, fading towards the bound.

    ``radius`` is ``entity.acceptance_radius()`` -- the same chi-square cutoff the
    condition containment test uses -- so the outermost shell is exactly the
    condition's boundary.  The shells are *surfaces* (no wireframe mesh), drawn
    small/opaque in the middle and large/faint at the bound: one flat colour per
    component, with the density in the alpha.  ``post`` halves the alpha and adds
    a faint outline, so pre and post of the same component stay distinguishable.
    """
    var = np.asarray(var, dtype=float)[:3]
    if var.ndim == 2:
        var = np.diag(var)[:3]
    std = np.sqrt(np.clip(var, 0.0, None))
    u, v = np.mgrid[0 : 2 * np.pi : 24j, 0 : np.pi : 16j]
    xyz = np.stack(
        [np.cos(u) * np.sin(v), np.sin(u) * np.sin(v), np.cos(v)], axis=-1
    )
    for fraction, alpha in ((0.45, 0.40), (0.75, 0.16), (1.0, 0.07)):
        r = n_std * radius * fraction
        pts = xyz * (r * std) + np.asarray(mean, dtype=float)[:3]
        ax.plot_surface(
            pts[..., 0], pts[..., 1], pts[..., 2],
            color=colour,
            alpha=alpha * (0.5 if post else 1.0),
            linewidth=0.25 if post else 0,
            edgecolor=colour if post else "none",
            antialiased=False, shade=False,
        )


def plot_entity(ax, entity, condition, label, limit, post=False):
    """One entity: every mixture component (a segment) as a shaded cloud."""
    p = entity.secure_mix_parameters(condition.models[label].get_parameters())
    means = p["measurement"]["pose"]["means"]
    covs = p["measurement"]["pose"]["covariances"]
    n = len(p["weights"])
    radius = entity.acceptance_radius()
    cmap = plt.get_cmap("tab10")  # one flat shade per component

    for k in range(n):
        colour = cmap(k % 10)
        shells(ax, means[k], covs[k], colour, radius, post=post)

    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)
    ax.set_zlim(-limit, limit)
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_zlabel("z [m]")
    ax.set_title(f"{label} ({n} component(s))", fontsize=9)
    ax.tick_params(labelsize=6)


# ============================================================================
#  2d view: TAPAS style, one panel per dimension, x = component index
# ============================================================================
# TAPAS dbg-prediction style: x/y/z share one axis, x=red y=green z=blue, and
# every component's Gaussian is an ellipse over the categorical component axis
DIM_COLORS = ("tab:red", "tab:green", "tab:blue")
GAUSSIAN_SCALE = 1.96  # ellipse spans +-1.96 sigma (95 %)
ELLIPSE_WIDTH = 0.6  # visual extent of one component on the categorical axis


def mix_params(entity, condition, label):
    """Component means/sigmas of the pose and the state distribution."""
    p = entity.secure_mix_parameters(condition.models[label].get_parameters())
    means = np.asarray(p["measurement"]["pose"]["means"], dtype=float)
    covs = np.asarray(p["measurement"]["pose"]["covariances"], dtype=float)
    if covs.ndim == 3:  # full covariance matrices
        variances = np.diagonal(covs, axis1=1, axis2=2)
    else:  # per-dimension variances already
        variances = covs
    sigmas = np.sqrt(np.clip(np.asarray(variances, dtype=float), 0.0, None))
    pis = np.asarray(p["measurement"]["state"]["pis"], dtype=float)
    return means, sigmas, pis


def extra_label(entity, dim):
    name = type(entity).__name__
    if "Prismatic" in name:
        return "slide (normalized)"
    if "Revolute" in name:
        return "angle [rad]"
    return f"extra [{dim}]"


def draw_gaussians(ax, means, sigmas, dim, colour, filled):
    """Every component's Gaussian as an ellipse over the component axis."""
    for k in range(len(means)):
        ellipse = Ellipse(
            (float(k), float(means[k][dim])),
            width=ELLIPSE_WIDTH,
            height=2.0 * GAUSSIAN_SCALE * float(sigmas[k][dim]),
            facecolor=colour if filled else "none",
            edgecolor=colour,
            alpha=0.30 if filled else 0.55,
            linewidth=1.0 if filled else 0.8,
            linestyle="-" if filled else "--",
            zorder=3 if filled else 2,
        )
        ax.add_patch(ellipse)


def to_metric(scene, means, sigmas, dims=3):
    """Scene-normalized positions -> metres (as the 3d view intends)."""
    center = np.asarray(getattr(scene, "_meta_xyz_center", np.zeros(3)), float).ravel()
    scaler = float(np.asarray(getattr(scene, "_meta_xyz_scaler", 1.0)).ravel()[0])
    means = means.copy()
    sigmas = sigmas.copy()
    for dim in range(min(dims, means.shape[1])):
        means[:, dim] = means[:, dim] / scaler + center[dim]
        sigmas[:, dim] = sigmas[:, dim] / scaler
    return means, sigmas


def plot_entity_axes(entity, label, pre, post, title, path, scene):
    """One figure for an entity: x/y/z, the extra dims, and the state."""
    pre_m, pre_s, pre_p = mix_params(entity, pre, label)
    post_m, post_s, post_p = mix_params(entity, post, label)
    pre_m, pre_s = to_metric(scene, pre_m, pre_s)
    post_m, post_s = to_metric(scene, post_m, post_s)
    pose_dim = pre_m.shape[1]
    rot = int(getattr(entity, "rot_dim", 0))
    extra_dims = list(range(3 + rot, pose_dim))
    n_comp = max(len(pre_m), len(post_m))
    n_states = max(pre_p.shape[1], post_p.shape[1])
    cmap = plt.get_cmap("tab10")

    fig = plt.figure(figsize=(13.5, 4.4))
    gs = fig.add_gridspec(1, 3, wspace=0.3)
    ax_pos = fig.add_subplot(gs[0, 0])
    ax_extra = fig.add_subplot(gs[0, 1])
    ax_state = fig.add_subplot(gs[0, 2])

    # x, y and z share one axes, each as its own Gaussian color
    for dim in range(min(3, pose_dim)):
        colour = DIM_COLORS[dim]
        draw_gaussians(ax_pos, pre_m, pre_s, dim, colour, filled=True)
        draw_gaussians(ax_pos, post_m, post_s, dim, colour, filled=False)
    ax_pos.set_xticks(range(n_comp))
    ax_pos.set_xlim(-0.5, n_comp - 0.5)
    ax_pos.set_xlabel("component", fontsize=8)
    ax_pos.set_ylabel("position [m]", fontsize=8)
    ax_pos.set_title("position", fontsize=9)
    ax_pos.tick_params(labelsize=6, direction="in", top=True, right=True)
    handles = [
        plt.Line2D([], [], color=DIM_COLORS[d], lw=2, label=name)
        for d, name in list(enumerate(("x", "y", "z")))[: min(3, pose_dim)]
    ]
    ax_pos.legend(handles=handles, frameon=False, fontsize=6, loc="upper right")

    if extra_dims:
        for j, dim in enumerate(extra_dims):
            colour = DIM_COLORS[j % len(DIM_COLORS)]
            draw_gaussians(ax_extra, pre_m, pre_s, dim, colour, filled=True)
            draw_gaussians(ax_extra, post_m, post_s, dim, colour, filled=False)
        ax_extra.set_xticks(range(n_comp))
        ax_extra.set_xlim(-0.5, n_comp - 0.5)
        ax_extra.set_xlabel("component", fontsize=8)
        ax_extra.set_ylabel(extra_label(entity, extra_dims[0]), fontsize=8)
        ax_extra.set_title("extra dims", fontsize=9)
        ax_extra.tick_params(labelsize=6, direction="in", top=True, right=True)
    else:
        ax_extra.text(0.5, 0.5, "no extra dims", ha="center", va="center",
                      transform=ax_extra.transAxes, fontsize=8)
        ax_extra.axis("off")

    if n_states > 1:
        for k in range(pre_p.shape[0]):
            for state in range(pre_p.shape[1]):
                ax_state.plot(
                    k - 0.15 + 0.25 * state,
                    float(pre_p[k][state]),
                    "o",
                    color=cmap(state % 10),
                    ms=3.5,
                )
        for k in range(post_p.shape[0]):
            for state in range(post_p.shape[1]):
                ax_state.plot(
                    k + 0.15 + 0.25 * state,
                    float(post_p[k][state]),
                    "s",
                    color=cmap(state % 10),
                    ms=3.0,
                    mfc="none",
                )
        ax_state.set_xlabel("component", fontsize=8)
        ax_state.set_ylabel("probability", fontsize=8)
        ax_state.set_title("state", fontsize=9)
        ax_state.set_xticks(range(n_comp))
        ax_state.set_xlim(-0.5, n_comp - 0.5)
        ax_state.set_ylim(-0.05, 1.05)
        ax_state.tick_params(labelsize=6, direction="in", top=True, right=True)
    else:
        ax_state.text(0.5, 0.5, "single state", ha="center", va="center",
                      transform=ax_state.transAxes, fontsize=8)
        ax_state.axis("off")

    fig.suptitle(f"{title} — {label} ({n_comp} component(s))", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {path}")


# ============================================================================
#  main
# ============================================================================
def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_scene_argument(ap)
    add_model_argument(ap)
    add_use_gt_argument(ap)
    ap.add_argument("--lim", type=float, default=5.0, help="axis limits, +-m (5).")
    ap.add_argument(
        "--style",
        choices=("both", "3d", "2d"),
        default="both",
        help="which view(s) to write: the 3d ellipsoid clouds, the per-axis 2d "
        "plots, or both (default both).",
    )
    ap.add_argument(
        "--conditions",
        choices=("pre", "post", "both"),
        default="both",
        help="Which conditions to draw: the segment start (pre), the segment end "
        "(post, fainter with an outline), or both (default).",
    )
    ap.add_argument("--out", default="", help="Output directory (default: <repo>).")
    args = ap.parse_args()

    scene = Scene.get(scene_config(args.scene))
    configs = [
        cfg
        for cfg in experts_for_scene(args.scene)
        if not (args.model and cfg.tag != args.model)
    ]
    if not configs:
        raise SystemExit(f"no expert model {args.model!r} in {args.scene}")

    out_dir = Path(args.out) if args.out else Path(_REPO_ROOT)
    for cfg in configs:
        model = ExpertModel.get(cfg, auto_load=False)
        model.use_gt(args.gt)
        model.load()
        pair = model.conditions
        entities = sorted(set(pair.pre.entities) & set(pair.post.entities))
        if not entities:
            logger.warning(f"{cfg.tag}: no shared entities in the conditions")
            continue

        if args.style in ("both", "3d"):
            figure = plt.figure(figsize=(4.6 * len(entities), 4.4))
            for index, label in enumerate(entities, start=1):
                axis = figure.add_subplot(1, len(entities), index, projection="3d")
                entity = pair.pre.entities[label]
                # `pre` holds the start conditions of each segment; `post` is the
                # matching end, drawn in the same axes so the segment motion shows
                if args.conditions in ("pre", "both"):
                    plot_entity(axis, entity, pair.pre, label, args.lim, post=False)
                if args.conditions in ("post", "both"):
                    plot_entity(axis, entity, pair.post, label, args.lim, post=True)
            figure.legend(loc="lower center", ncol=4, fontsize=7, frameon=False)
            figure.tight_layout()
            path = out_dir / f"conditions_{cfg.tag}_3d.png"
            figure.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(figure)
            logger.info(f"wrote {path}")

        if args.style in ("both", "2d"):
            for label in entities:
                entity = pair.pre.entities[label]
                path = out_dir / f"conditions_{cfg.tag}_{label}_axes.png"
                plot_entity_axes(
                    entity, label, pair.pre, pair.post, cfg.tag, path, scene
                )


if __name__ == "__main__":
    main()
