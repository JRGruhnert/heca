"""Violin plots of the visual pipeline against ground truth, per entity.

Positions: the frame each reference was clicked in gives one fixed offset between
the keypoint read and the ground truth there. Every sampled frame is corrected by
that offset, and what remains between the corrected read and that frame's own
ground truth is the error -- so the plot shows the visual pipeline's error, not the
frame the visual pipeline happens to use.

States: the state encoder's confidence in the ground-truth answer, 0 where it
answered wrong, plus the accuracy it reached.

    python -m scripts.d07_error_plots --scene scene0 --samples 100 --kp-backend roma
    python -m scripts.d07_error_plots --scene scene0 --samples 20 --no-states
"""

import argparse
import json
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heca.data.reference import Reference
from heca.experts.expert import ExpertModel
from heca.misc import logger
from heca.scenes.scene import Scene
from scripts.common.args import (
    add_encoder_arguments,
    add_scene_argument,
    add_threads_argument,
)
from scripts.common.scenes import experts_for_scene, scene_config, select_encoder

IMAGE_KEYS = {"rgb", "depth", "mask", "extrinsics", "intrinsics", "demo"}


def frame_at(file: h5py.File, row: int, scene: Scene):
    image = {key: file[key][row] for key in IMAGE_KEYS if key != "demo"}
    extra = {key: file[key][row] for key in file.keys() if key not in IMAGE_KEYS}
    obs, _ = scene.to_internal(image, extra)
    return scene.to_td_image(obs)


def darker(color, factor: float = 0.55):
    """A shade of ``color`` for the marks drawn on top of it."""
    return tuple(factor * channel for channel in color[:3]) + tuple(color[3:] or (1.0,))


def violin(groups: dict[str, list[float]], ylabel: str, title: str, out: Path):
    """One violin per entity: body in the entity's colour, marks in a darker shade.

    Every group is a distance to the ground truth, so zero is the smallest value
    either distribution can take and the y axis runs from there to the largest.
    """
    names = [name for name, values in groups.items() if len(values) > 1]
    data = [np.asarray(groups[name], dtype=float) for name in names]
    if not data:
        logger.warning(f"{out.name}: nothing to plot")
        return
    palette = plt.get_cmap("tab10")
    colors = [palette(index % 10) for index in range(len(names))]
    body_width = 0.6  # narrower than the default 0.8, so the columns stay apart
    mark_half = 0.12  # half-length of the min / median / max marks
    figure, axis = plt.subplots(figsize=(1.5 * len(names) + 2.0, 4.6))
    parts = axis.violinplot(
        data,
        positions=range(1, len(names) + 1),
        widths=body_width,
        showextrema=False,
    )
    for body, color in zip(parts["bodies"], colors):
        body.set_facecolor(color)
        body.set_edgecolor(darker(color))
        body.set_alpha(0.55)
        body.set_linewidth(1.0)

    spread = np.random.default_rng(0)
    for index, (values, color) in enumerate(zip(data, colors), start=1):
        shade = darker(color)
        axis.scatter(
            index + spread.uniform(-mark_half, mark_half, len(values)),
            values,
            s=6,
            color=shade,
            alpha=0.5,
            linewidths=0,
        )
        # min, median and max drawn across the entity's column
        for value, style in (
            (0.0, (0, (4, 3))),
            (float(np.median(values)), "-"),
            (float(values.max()), (0, (2, 2))),
        ):
            axis.plot(
                [index - mark_half, index + mark_half],
                [value, value],
                color=shade,
                linewidth=1.6,
                linestyle=style,
            )

    axis.set_xticks(range(1, len(names) + 1), names)
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel(ylabel)
    # axis.set_title(title)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=150)
    plt.close(figure)
    logger.info(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    add_threads_argument(parser)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        default="",
        help="Directory for the figures (default: the scene's own folder).",
    )
    parser.add_argument(
        "--no-states", action="store_true", help="Skip the state encoder (no Molmo)."
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-measure even when a cache of the measurements exists.",
    )
    args = parser.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    scene_cfg = scene_config(args.scene)
    out_dir = Path(args.out) if args.out else Path(Scene.save_dir(scene_cfg))
    cache = out_dir / f"measurements-{args.kp_backend}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else None
    fresh = (
        cached is not None
        and cached.get("samples") == args.samples
        and cached.get("seed") == args.seed
        and cached.get("with_states") == (not args.no_states)
        and not args.refresh
    )
    if fresh:
        logger.info(f"{cache.name}: reusing the cached measurements")
        payload = cached
    else:
        select_encoder(args.scene, args.backbone, args.stride, args.kp_backend)
        scene = Scene.get(scene_config(args.scene))
        model = ExpertModel.get(experts_for_scene(args.scene)[0], auto_load=False)
        model.use_gt(False)
        model.use_states = not args.no_states
        model.load()
        labels = list(scene.entities)
        out_dir = Path(args.out) if args.out else Path(scene.save_dir(scene.cfg))

        root = Path(scene.save_dir(scene.cfg))
        candidates = [path for path in sorted(root.glob("*.h5"))]
        dataset = None
        for path in candidates:
            with h5py.File(path, "r") as probe:
                if "rgb" in probe and f"heca_{labels[0]}_pos" in probe:
                    dataset = path
                    break
        if dataset is None:
            raise SystemExit(f"no dataset with frames and ground truth in {root}")
        logger.info(f"{dataset.name}: frames and ground truth")

        with h5py.File(dataset, "r") as file:
            total = len(file["rgb"])

            offsets: dict[str, np.ndarray] = {}
            for label in labels:
                reference = scene.references[label][Reference.POSITION]
                frame = frame_at(file, reference.frame, scene)
                model.reset_tracking()
                keypoint = model.kp_extractor.extract_poses(frame)[label]
                truth = np.asarray(file[f"heca_{label}_pos"][reference.frame], dtype=float)
                offsets[label] = truth - keypoint.xyz
                logger.info(
                    f"{label}: reference offset {np.round(offsets[label], 3)} m "
                    f"(read {np.round(keypoint.xyz, 3)})"
                )

            rows = np.sort(
                np.random.default_rng(args.seed).choice(total, args.samples, replace=False)
            )
            errors: dict[str, list[float]] = {label: [] for label in labels}
            scores: dict[str, list[float]] = {label: [] for label in labels}
            correct: dict[str, int] = {label: 0 for label in labels}
            for done, row in enumerate(rows, start=1):
                frame = frame_at(file, row, scene)
                model.reset_tracking()
                poses = model.kp_extractor.extract_poses(frame)
                states = (
                    model.ste_extractor.extract_states(frame) if model.use_states else {}
                )
                for label in labels:
                    truth = np.asarray(file[f"heca_{label}_pos"][row], dtype=float)
                    corrected = poses[label].xyz + offsets[label]
                    errors[label].append(float(np.linalg.norm(corrected - truth)))
                    if label in states:
                        index, probability = states[label]
                        expected = int(np.ravel(file[f"heca_{label}_ste"][row])[0])
                        hit = index == expected
                        correct[label] += int(hit)
                        scores[label].append(float(probability) if hit else 0.0)
                if done % 10 == 0 or done == len(rows):
                    logger.info(f"{done}/{len(rows)} frames read")
        payload = {
            "scene": args.scene,
            "backend": args.kp_backend,
            "samples": args.samples,
            "seed": args.seed,
            "with_states": not args.no_states,
            "labels": labels,
            "errors": errors,
            "scores": scores,
            "correct": correct,
        }
        cache.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(f"cached the measurements to {cache}")

    errors = payload["errors"]
    scores = payload["scores"]
    correct = payload["correct"]
    labels = payload["labels"]

    for label in labels:
        values = np.asarray(errors[label])
        logger.info(
            f"{label:9s} position error mean {values.mean():.3f} "
            f"median {np.median(values):.3f} max {values.max():.3f} m"
        )
    violin(
        errors,
        "distance to the ground-truth position [m]",
        f"{args.scene} / {args.kp_backend}: visual keypoint vs ground truth",
        out_dir / f"position-error-{args.kp_backend}.png",
    )

    if scores:
        for label in labels:
            if scores[label]:
                logger.info(
                    f"{label:9s} state accuracy {correct[label] / len(scores[label]):.2f}"
                )
        violin(
            scores,
            "state confidence for the ground-truth answer (0 = wrong)",
            f"{args.scene}: state readout vs ground truth",
            out_dir / f"state-score-{args.kp_backend}.png",
        )


if __name__ == "__main__":
    main()
