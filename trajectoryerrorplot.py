import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

# ============================================================================
#  data
# ============================================================================
CSV_DIR = "visual"  # folder holding <model>_<backend>_errors.csv
CSV_GLOB = "*_errors.csv"
OUT_DIR = "out/errors"
SCENE = "scene0"
ERROR_COLUMN = "error_offset"  # b06/b07 output; "error" is the fallback
GRID_POINTS = 100  # points per segment of the normalized-step axis
SMOOTH_FRACTION = 0.05  # gaussian sigma as a fraction of the segment length

# Axis
DPI = 300  # 300 for papers, 150 for quick looks
XMIN = 0.0  # every segment is normalized to its own unit interval
YMIN = -0.01  # set to None to fit the data
YMAX = None  # set to None to fit mean+std with padding
PAD = 0.05  # fractional padding when a limit is None
# Figure
FIG_SIZE_CM = (9.0, 6.0)  # same default as the .sty
LINE_WIDTH = 0.5
SEGMENT_BOUNDARY_LINE_WIDTH = 0.4  # vertical dotted segment separators
SEGMENT_BOUNDARY_ALPHA = 0.8
SEGMENT_BOUNDARY_COLOR = "0.55"
SEGMENT_BOUNDARY_LINESTYLE = ":"
BAND_ALPHA = 0.40  # 0.20 ~= xcolor "!20"
SHOW_BAND = True  # False -> mean lines only
LEGEND_LOC = "lower right"  # matplotlib "loc" keyword
GRID = False
COLORS = [
    "hotpink",
    "springgreen",
    "cyan",
    "steelblue",
    "darkmagenta",
    "darkorange",
    "red",
    "saddlebrown",
    "olive",
    "yellow",
]
FONT_SIZE = 8  # base / default text
LABEL_SIZE = 9  # axis labels ("Epoch", "Success rate")
TICK_SIZE = 7  # tick labels on both axes
LEGEND_SIZE = 7  # legend entries
TITLE_SIZE = 9  # axes title

XLABEL = "Normalized step"
YLABEL = "Trajectory error [m]"
YLABEL_AXIS = "Trajectory {axis} error [m]"


# ============================================================================
#  implementation (no need to touch below)
# ============================================================================
def num(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_rows(
    path: Path,
) -> tuple[dict[str, dict[int, dict[int, dict[str, float]]]], str, bool]:
    """Per entity, per demo: ``{"total", "x", "y", "z"}`` error per step.

    ``total`` is the error column (the offset-corrected one when present, else
    ``error``); the per-axis values are ``|vis_<axis> - gt_<axis>|`` in the same
    units (metres).
    """
    with open(path, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        sys.exit(f"error: {path!r} is empty")
    header = rows[0].keys()
    if ERROR_COLUMN not in header and "error" not in header:
        sys.exit(
            f"error: neither {ERROR_COLUMN!r} nor 'error' in {path!r}\n"
            f"       available: {', '.join(header)}"
        )
    column = ERROR_COLUMN if ERROR_COLUMN in header else "error"
    has_axes = all(
        key in header
        for key in ("gt_x", "gt_y", "gt_z", "vis_x", "vis_y", "vis_z")
    )
    # b07 writes the offset-corrected per-axis columns; the raw |vis - gt| is
    # only the fallback for CSVs that were not corrected yet
    axis_columns = {
        axis: f"error_offset_{axis}" for axis in ("x", "y", "z")
    }
    axis_columns = {
        axis: name for axis, name in axis_columns.items() if name in header
    }

    has_segment = "segment" in header and "segment_t" in header

    per_entity: dict[str, dict[int, dict[int, dict[str, float]]]] = {}
    for row in rows:
        entity = str(row.get("entity", ""))
        demo = int(num(row["demo"]))
        step = int(num(row["timestep"]))
        values = {"total": num(row[column])}
        if has_axes:
            for axis in ("x", "y", "z"):
                value = num(row[axis_columns[axis]]) if axis in axis_columns else float("nan")
                if np.isnan(value):
                    value = abs(num(row[f"vis_{axis}"]) - num(row[f"gt_{axis}"]))
                values[axis] = value
        values["segment"] = num(row["segment"]) if has_segment else float("nan")
        values["segment_t"] = num(row["segment_t"]) if has_segment else float("nan")
        per_entity.setdefault(entity, {}).setdefault(demo, {})[step] = values
    return per_entity, column, has_axes, has_segment


def model_expert(model_tag: str):
    """The loaded expert of a model (its fitted frames and demo segmentation)."""
    try:
        from heca.experts.expert import ExpertModel
        from scripts.common.scenes import experts_for_scene

        cfg = next((c for c in experts_for_scene(SCENE) if c.tag == model_tag), None)
        if cfg is None:
            return None, None
        return cfg, ExpertModel.get(cfg)
    except Exception as exc:  # missing, not fitted, ...
        print(f"{model_tag}: cannot load expert ({exc})")
        return None, None


def task_entities(expert) -> list[str] | None:
    """The task parameters of the model: the frames its tapas model is fitted on."""
    if type(expert).__name__ != "TapasExpert":
        return None
    try:
        return sorted(expert.tps)
    except Exception:
        return None


def split_model_tag(stem: str) -> str:
    """``drawer0_a_b_dino`` -> ``drawer0_a_b``."""
    for backend in ("dino", "roma"):
        suffix = f"_{backend}"
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def demo_segments(expert, demo_ids: list[int]) -> dict[int, list[tuple[int, int]]]:
    """The velocity segmentation of every demo, as frame ranges.

    The boundaries are the same the fit uses (``_velocity_segments``): the demo
    is split where the end-effector velocity pauses, so ``[(0, 41), (41, 96)]``
    means the first segment covers frames 0..40 and the second 41..95.
    """
    import h5py

    from heca.experts.tapas import TapasExpert

    if not isinstance(expert, TapasExpert):
        return {}
    seg_cfg = expert.cfg.policy.model.demos_segmentation
    demos_path = expert.load_dir(expert.cfg) / "demos.h5"
    if not demos_path.exists():
        return {}
    segments: dict[int, list[tuple[int, int]]] = {}
    with h5py.File(demos_path, "r") as demos:
        for demo_id in demo_ids:
            try:
                scenes, _ = expert.scene.load_dataset(
                    demos, selections=[demo_id], with_images=False
                )
                td = expert.dcscenes_to_tdtapas(scenes[0])
                vel = np.linalg.norm(td.action[:, :3].numpy().astype(float), axis=1)
            except Exception:
                continue
            boundaries = expert._velocity_segments(vel, seg_cfg)
            edges = [0, *boundaries, len(vel)]
            ranges = [
                (edges[i], edges[i + 1])
                for i in range(len(edges) - 1)
                if edges[i + 1] > edges[i]
            ]
            segments[demo_id] = ranges or [(0, len(vel))]
    return segments


def smooth(values: np.ndarray, fraction: float) -> np.ndarray:
    """Gaussian smoothing with a sigma proportional to the curve length."""
    n = len(values)
    if n < 3 or fraction <= 0:
        return values
    sigma = max(1.0, fraction * n)
    radius = max(1, int(np.ceil(4.0 * sigma)))
    offsets = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (offsets / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(values, radius, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def segment_curve_t(
    t_values: np.ndarray, errors: np.ndarray, grid: np.ndarray
) -> np.ndarray | None:
    """A segment curve from an already normalized in-segment time axis."""
    order = np.argsort(t_values)
    t = t_values[order]
    e = errors[order]
    if len(t) < 2 or t[-1] <= t[0]:
        return None
    return np.interp(grid, t, smooth(e, SMOOTH_FRACTION))


def segment_curve(
    steps: np.ndarray,
    errors: np.ndarray,
    seg_start: int,
    seg_end: int,
    grid: np.ndarray,
) -> np.ndarray | None:
    """One segment's error as a function of its own normalized step (0..1)."""
    mask = (steps >= seg_start) & (steps < seg_end)
    if mask.sum() < 2:
        return None
    seg_steps = steps[mask]
    span = float(seg_steps[-1] - seg_steps[0])
    if span <= 0:
        return None
    t = (seg_steps - seg_steps[0]) / span
    return np.interp(grid, t, smooth(errors[mask], SMOOTH_FRACTION))


def plot(
    title: str,
    kind: str,
    axis_labels: bool,
    grid: np.ndarray,
    segments: dict[str, dict[int, dict[str, np.ndarray]]],
    n_segments: int,
):
    """One figure per model: the x axis is the segment chain 0..n_segments.

    Each segment k occupies [k, k+1] with its own normalized step (0..1); every
    segment has its own mean +- std error over the demos, and the segment
    boundaries are drawn as vertical dotted lines.
    """
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    fig_w = FIG_SIZE_CM[0] / 2.54
    fig_h = FIG_SIZE_CM[1] / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_lo, y_hi = [], []
    for idx, (entity, per_segment) in enumerate(sorted(segments.items())):
        color = COLORS[idx % len(COLORS)]
        label = entity
        for k in sorted(per_segment):
            curves = per_segment[k]
            if not curves:
                continue
            stacked = np.vstack(curves)
            mean = np.nanmean(stacked, axis=0)
            if stacked.shape[0] > 1:
                with np.errstate(invalid="ignore"):
                    std = np.nanstd(stacked, axis=0, ddof=1)  # sample std (n-1)
                std = np.nan_to_num(std, nan=0.0)
            else:
                std = np.zeros(len(grid))
            x = k + grid
            if SHOW_BAND and np.any(std > 1e-12):
                ax.fill_between(
                    x,
                    np.maximum(mean - std, 0.0),  # error cannot go below 0
                    mean + std,
                    color=color,
                    alpha=BAND_ALPHA,
                    linewidth=0,
                )
                y_lo.append(float(np.nanmin(mean - std)))
                y_hi.append(float(np.nanmax(mean + std)))
            else:
                y_lo.append(float(np.nanmin(mean)))
                y_hi.append(float(np.nanmax(mean)))
            ax.plot(
                x,
                mean,
                color=color,
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
                label=label,
            )
            label = None  # one legend entry per entity

    for k in range(1, n_segments):
        ax.axvline(
            x=k,
            color=SEGMENT_BOUNDARY_COLOR,
            linewidth=SEGMENT_BOUNDARY_LINE_WIDTH,
            linestyle=SEGMENT_BOUNDARY_LINESTYLE,
            alpha=SEGMENT_BOUNDARY_ALPHA,
        )

    lo, hi = min(y_lo), max(y_hi)
    span = hi - lo if hi - lo > 1e-12 else 1.0
    ax.set_xlim(XMIN, float(n_segments))
    ax.set_ylim(
        YMIN if YMIN is not None else lo - PAD * span,
        YMAX if YMAX is not None else hi + PAD * span,
    )

    if axis_labels:
        ax.set_xlabel(XLABEL)
        ax.set_ylabel(
            YLABEL if kind == "total" else YLABEL_AXIS.format(axis=kind)
        )
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(GRID)
    ax.legend(loc=LEGEND_LOC, frameon=False)
    fig.tight_layout()
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)
    suffix = "" if kind == "total" else f"_{kind}"
    labeled = "_labeled" if axis_labels else ""
    output_path = Path(OUT_DIR) / f"{title}_trajectoryerror{suffix}{labeled}.png"
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {output_path}")


def curves_for_model(
    path: Path,
) -> tuple[dict[str, dict[str, dict[int, list[np.ndarray]]]], int]:
    """Per error kind, per entity, per segment: the normalized demo curves."""
    model_stem = path.stem.removesuffix("_errors")
    model_tag = split_model_tag(model_stem)
    per_entity, column, has_axes, has_segment = read_rows(path)
    print(f"{model_stem}: column {column!r}")

    _, expert = model_expert(model_tag)
    wanted = task_entities(expert) if expert is not None else None
    if wanted:
        entities = [e for e in wanted if e in per_entity]
        print(f"{model_stem}: task parameters {wanted}")
    else:
        entities = sorted(per_entity)
        print(f"{model_stem}: no task parameters, using {entities}")

    grid = np.linspace(0.0, 1.0, GRID_POINTS)
    kinds = ["total", "x", "y", "z"] if has_axes else ["total"]
    curves: dict[str, dict[str, dict[int, list[np.ndarray]]]] = {
        kind: {} for kind in kinds
    }
    n_segments = 1

    if has_segment:
        # the CSV already carries the segmentation (segment + normalized
        # in-segment time): use it as is
        print(f"{model_stem}: using the CSV segmentation (segment, segment_t)")
        for entity in entities:
            per_kind = {kind: {} for kind in kinds}
            for demo, steps in per_entity[entity].items():
                groups: dict[int, list[int]] = {}
                for step, values in steps.items():
                    seg = values["segment"]
                    if np.isnan(seg):
                        continue
                    groups.setdefault(int(seg), []).append(step)
                for seg, seg_steps in groups.items():
                    seg_steps = sorted(seg_steps, key=lambda s: steps[s]["segment_t"])
                    t_values = np.asarray(
                        [steps[s]["segment_t"] for s in seg_steps], dtype=float
                    )
                    n_segments = max(n_segments, seg + 1)
                    for kind in kinds:
                        errors = np.asarray(
                            [steps[s][kind] for s in seg_steps], dtype=float
                        )
                        curve = segment_curve_t(t_values, errors, grid)
                        if curve is not None:
                            per_kind[kind].setdefault(seg, []).append(curve)
            for kind in kinds:
                curves[kind][entity] = per_kind[kind]
        return curves, n_segments

    demo_ids = sorted({demo for demos in per_entity.values() for demo in demos})
    demo_ranges = demo_segments(expert, demo_ids) if expert is not None else {}
    for demo in demo_ids:
        ranges = demo_ranges.get(demo)
        if ranges:
            spans = ", ".join(f"{a}-{b - 1}" for a, b in ranges)
            print(f"  demo {demo}: {len(ranges)} segment(s): {spans}")
        else:
            print(f"  demo {demo}: no segmentation, one segment")

    for entity in entities:
        per_kind = {kind: {} for kind in kinds}
        for demo, steps in per_entity[entity].items():
            ordered = np.asarray(sorted(steps), dtype=float)
            ranges = demo_ranges.get(demo) or [(int(ordered[0]), int(ordered[-1]) + 1)]
            n_segments = max(n_segments, len(ranges))
            for kind in kinds:
                errors = np.asarray(
                    [steps[s][kind] for s in sorted(steps)], dtype=float
                )
                for k, (seg_start, seg_end) in enumerate(ranges):
                    curve = segment_curve(ordered, errors, seg_start, seg_end, grid)
                    if curve is not None:
                        per_kind[kind].setdefault(k, []).append(curve)
        for kind in kinds:
            curves[kind][entity] = per_kind[kind]
    return curves, n_segments


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--dir",
        type=Path,
        default=Path(CSV_DIR),
        help=f"folder with the error CSVs (default {CSV_DIR})",
    )
    ap.add_argument("--glob", default=CSV_GLOB)
    args = ap.parse_args()

    files = sorted(args.dir.glob(args.glob))
    if not files:
        sys.exit(f"error: no {args.glob!r} in {args.dir!r}")

    for path in files:
        title = path.stem.removesuffix("_errors")
        curves, n_segments = curves_for_model(path)
        grid = np.linspace(0.0, 1.0, GRID_POINTS)
        for kind, segments in curves.items():
            if not any(any(seg.values()) for seg in segments.values()):
                continue
            plot(title, kind, True, grid, segments, n_segments)
            plot(title, kind, False, grid, segments, n_segments)


if __name__ == "__main__":
    main()
