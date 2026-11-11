"""Per-entity error curves of one TAPAS model, normalised per segment.

Reads the ``<tag>_<backend>_errors.csv`` files written by
``b06_visual_trajectory_errors`` and draws, for one model, **one line per
entity** -- no averaging over entities: the entities are the series and the
model name lives in the file name, so the legend lists entities only.

Segments come from the fit's own velocity segmentation, recomputed from the
ground-truth demo scenes (``dcscenes_to_tdtapas`` + ``_velocity_segments``); no
image encoder is involved. Within every segment the error is normalised the way
TAPAS expresses a segment -- in the frame at its start:

    offset(segment, entity) = gt - vis   at the segment's first frame
    error_segment           = |vis + offset - gt|

so each segment starts at 0 and the curve shows how the read drifts inside the
segment. ``segment_t`` runs 0 -> 1 across each segment, which is the x axis by
default (``--x segment_t``), so segments of different lengths are comparable.

    python -m scripts.d10_plot_model_errors --backend dino
    python -m scripts.d10_plot_model_errors --tag drawer0_a_b --x timestep
"""

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import h5py

from heca.experts.expert import ExpertModel
from heca.experts.tapas import TapasExpert
from heca.misc import logger
from heca.misc.paths import find_repo_root
from heca.scenes.scene import Scene

from scripts.common.scenes import experts_for_scene, scene_config


def segments_for(scene: Scene, model_tag: str, demo_ids: list[int]) -> dict[int, np.ndarray]:
    """demo id -> per-frame segment index, from the fit's velocity segmentation."""
    cfg = next(c for c in experts_for_scene(scene.cfg.tag) if c.tag == model_tag)
    expert = ExpertModel.get(cfg, auto_load=False)
    expert.use_states = False
    expert.use_extra = False
    expert.use_gt(True)  # GT scenes: the segmentation never reads an image
    seg_cfg = cfg.policy.model.demos_segmentation
    path = TapasExpert.load_dir(cfg) / "demos.h5"
    out: dict[int, np.ndarray] = {}
    with h5py.File(path, "r") as file:
        for demo_id in demo_ids:
            scenes, _ = scene.load_dataset(
                file, selections=[demo_id], with_scenes=True, with_images=False
            )
            td = expert.dcscenes_to_tdtapas(scenes[0])
            vel = np.linalg.norm(td.action[:, :3].numpy().astype(float), axis=1)
            boundaries = expert._velocity_segments(vel, seg_cfg)
            out[demo_id] = np.searchsorted(np.asarray(boundaries), np.arange(len(vel)))
    return out


def read_csv(path: Path) -> list[dict]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def normalise(rows: list[dict], segments: dict[int, np.ndarray]) -> list[dict]:
    """Add ``segment``, ``segment_t`` and ``error_segment`` to every row."""
    by_demo: dict[int, list[dict]] = {}
    for row in rows:
        by_demo.setdefault(int(row["demo"]), []).append(row)

    out: list[dict] = []
    for demo_id, demo_rows in by_demo.items():
        seg = segments.get(demo_id)
        if seg is None:
            logger.warning(f"demo {demo_id}: no segmentation, rows skipped")
            continue
        demo_rows.sort(key=lambda r: int(r["timestep"]))
        for row in demo_rows:
            step = int(row["timestep"])
            row["segment"] = int(seg[step]) if step < len(seg) else int(seg[-1])
        by_segment: dict[int, list[dict]] = {}
        for row in demo_rows:
            by_segment.setdefault(row["segment"], []).append(row)
        for segment, seg_rows in by_segment.items():
            first = seg_rows[0]
            span = max(1, len(seg_rows) - 1)
            for index, row in enumerate(seg_rows):
                truth = np.array([float(row[f"gt_{axis}"]) for axis in "xyz"])
                seen = np.array([float(row[f"vis_{axis}"]) for axis in "xyz"])
                anchor_truth = np.array([float(first[f"gt_{axis}"]) for axis in "xyz"])
                anchor_seen = np.array([float(first[f"vis_{axis}"]) for axis in "xyz"])
                offset = anchor_truth - anchor_seen
                row["segment_t"] = index / span
                row["error_segment"] = float(np.linalg.norm(seen + offset - truth))
                row["error_norm_start"] = float(
                    np.linalg.norm(seen - truth) - np.linalg.norm(anchor_seen - anchor_truth)
                )
                out.append(row)
    return out


def plot(tag: str, backend: str, rows: list[dict], column: str, x_key: str, out: Path) -> None:
    entities = sorted({row["entity"] for row in rows})
    palette = plt.get_cmap("tab10")
    figure, axis = plt.subplots(figsize=(0.9 * len(entities) + 4.0, 3.6))
    bins = 40

    for index, entity in enumerate(entities):
        colour = palette(index % 10)
        selected = [row for row in rows if row["entity"] == entity]
        xs = np.asarray([float(row[x_key]) for row in selected])
        ys = np.asarray([float(row[column]) for row in selected])
        order = np.argsort(xs)
        xs, ys = xs[order], ys[order]
        if x_key == "segment_t":
            # 0 -> 1 per segment: bin over the normalised time and average
            edges = np.linspace(0.0, 1.0, bins + 1)
            centre = 0.5 * (edges[:-1] + edges[1:])
            index_of = np.clip(np.digitize(xs, edges) - 1, 0, bins - 1)
            means = np.full(bins, np.nan)
            for cell in range(bins):
                values = ys[index_of == cell]
                if len(values):
                    means[cell] = values.mean()
            axis.plot(centre, means, color=colour, linewidth=1.6, label=entity)
        else:
            # one faint line per demo, a bold mean across demos
            by_demo: dict[int, list[tuple[float, float]]] = {}
            for row in selected:
                by_demo.setdefault(int(row["demo"]), []).append(
                    (float(row[x_key]), float(row[column]))
                )
            grid = np.linspace(xs.min(), xs.max(), bins)
            stack = []
            for demo_rows in by_demo.values():
                px = np.asarray([p[0] for p in demo_rows])
                py = np.asarray([p[1] for p in demo_rows])
                order = np.argsort(px)
                axis.plot(px[order], py[order], color=colour, alpha=0.25, linewidth=0.7)
                stack.append(np.interp(grid, px[order], py[order]))
            axis.plot(grid, np.mean(stack, axis=0), color=colour, linewidth=1.6, label=entity)

    axis.set_xlabel("normalised time in segment" if x_key == "segment_t" else "timestep")
    axis.set_ylabel(column.replace("_", " "))
    axis.set_ylim(bottom=0.0)
    axis.grid(alpha=0.3)
    axis.legend(title=None, frameon=False)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(figure)
    logger.info(f"wrote {out} ({len(entities)} entities, {len(rows)} rows)")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--tag", default="", help="Only csv files whose name contains this.")
    parser.add_argument(
        "--backend", choices=("dino", "roma", "all"), default="all"
    )
    parser.add_argument("--dir", default="", help="Directory of the csv files (default <repo>/visual).")
    parser.add_argument(
        "--column",
        default="error_segment",
        choices=("error", "error_offset", "error_segment", "error_norm_start"),
    )
    parser.add_argument("--x", default="segment_t", choices=("segment_t", "timestep"))
    parser.add_argument("--out", default="", help="Output directory (default: same as --dir).")
    args = parser.parse_args()

    root = Path(args.dir) if args.dir else find_repo_root() / "visual"
    out_dir = Path(args.out) if args.out else root
    backends = ["dino", "roma"] if args.backend == "all" else [args.backend]
    scene = Scene.get(scene_config(args.scene))

    files = []
    for backend in backends:
        files += sorted(root.glob(f"*_{backend}_errors.csv"))
    if args.tag:
        files = [p for p in files if args.tag in p.name]
    files = [p for p in files if ".pre_offset" not in p.name]
    if not files:
        raise SystemExit(f"no *_errors.csv in {root} for backends {backends}")

    for path in files:
        model_tag, _, backend = path.name.replace("_errors.csv", "").rpartition("_")
        rows = read_csv(path)
        if not rows:
            logger.warning(f"{path.name}: empty, skipped")
            continue
        demo_ids = sorted({int(row["demo"]) for row in rows})
        segments = segments_for(scene, model_tag, demo_ids)
        rows = normalise(rows, segments)
        if not rows:
            continue
        plot(
            model_tag,
            backend,
            rows,
            args.column,
            args.x,
            out_dir / f"{path.stem}_by_entity.png",
        )


if __name__ == "__main__":
    main()
