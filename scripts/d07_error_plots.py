"""Violin plots of the visual pipeline against ground truth, per entity.

Positions: the frame each reference was clicked in gives one fixed offset between
the keypoint read and the ground truth there. Every sampled frame is corrected by
that offset, and what remains between the corrected read and that frame's own
ground truth is the error -- so the plot shows the visual pipeline's error, not the
frame the visual pipeline happens to use.

States: the state encoder's confidence in the ground-truth answer, 0 where it
answered wrong, plus the accuracy it reached.

    python -m scripts.d07_error_plots --scene scene0            # both backends, cached
    python -m scripts.d07_error_plots --scene scene0 --resample # fresh, 256 samples
    python -m scripts.d07_error_plots --scene scene0 --kp-backend roma --no-states
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
from scipy.stats import gaussian_kde

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


def darker(color, factor: float = 0.50):
    """A shade of ``color`` for the marks drawn on top of it."""
    return tuple(factor * channel for channel in color[:3]) + tuple(color[3:] or (1.0,))


def violin(groups: dict[str, list[float]], out: Path):
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
    body_width = 0.75  # narrower than the default 0.8, so the columns stay apart
    mark_half = 0.05  # half-length of the min / median / max marks
    figure, axis = plt.subplots(figsize=(1.0 * len(names) + 1.0, 3.8))

    for index, (values, color) in enumerate(zip(data, colors), start=1):
        if float(np.std(values)) <= 0.0:
            axis.plot(
                [index - body_width / 2.0, index + body_width / 2.0],
                [float(values[0]), float(values[0])],
                color=color,
                alpha=0.55,
                linewidth=1.0,
            )
            continue
        kde = gaussian_kde(values)
        bandwidth = float(kde.factor * np.std(values))
        # support min-2*bandwidth .. max: the same Scott KDE as violinplot, but
        # extended two bandwidths below the smallest sample (clipped at 0) so the
        # lower tail tapers instead of being cut off at the data minimum
        lo = max(0.0, float(values.min()) - 2.0 * bandwidth)
        grid = np.linspace(lo, float(values.max()), 200)
        density = kde(grid)
        density = body_width / 2.0 * density / density.max()
        axis.fill_betweenx(
            grid, index - density, index + density, color=color, alpha=0.55
        )
        axis.plot(index - density, grid, color=darker(color), linewidth=1.0)
        axis.plot(index + density, grid, color=darker(color), linewidth=1.0)

    for index, (values, color) in enumerate(zip(data, colors), start=1):
        shade = darker(color)
        axis.scatter(
            np.full(len(values), float(index)),
            values,
            s=6,
            color=shade,
            alpha=0.5,
            linewidths=0,
        )
        # min, median and max drawn across the entity's column
        for value in (0.0, float(np.median(values)), float(values.max())):
            axis.plot(
                [index - mark_half, index + mark_half],
                [value, value],
                color=shade,
                linewidth=1.5,
                linestyle="-",
            )

    axis.set_xticks(range(1, len(names) + 1), names)
    axis.set_ylim(bottom=0.0, top=0.7)
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(out, dpi=300)
    plt.close(figure)
    logger.info(f"wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    for action in parser._actions:
        if action.dest == "kp_backend":
            action.choices = ("dino", "roma", "all")
            action.default = "all"
    add_threads_argument(parser)
    parser.add_argument("--samples", type=int, default=256)
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
        "--resample",
        action="store_true",
        help="Re-measure instead of reusing a cached measurement.",
    )
    args = parser.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    scene_cfg = scene_config(args.scene)
    out_dir = Path(args.out) if args.out else Path(Scene.save_dir(scene_cfg))
    backends = ["dino", "roma"] if args.kp_backend == "all" else [args.kp_backend]
    for backend in backends:
        plot_backend(backend, args, out_dir)


def plot_backend(backend: str, args, out_dir: Path) -> None:
    cache = out_dir / f"measurements-{backend}.json"
    cached = json.loads(cache.read_text()) if cache.exists() else None
    if cached and cached.get("seed") == args.seed and not args.resample:
        logger.info(f"{cache.name}: reusing the cached measurements")
        payload = cached
    else:
        payload = measure(backend, args, out_dir, cache)

    errors = payload["errors"]
    scores = payload["scores"]
    correct = payload["correct"]
    labels = payload["labels"]

    for label in labels:
        values = np.asarray(errors[label])
        logger.info(
            f"[{backend}] {label:9s} position error mean {values.mean():.3f} "
            f"median {np.median(values):.3f} max {values.max():.3f} m"
        )
    violin(
        errors,
        out_dir / f"position-error-{backend}.png",
    )

    if scores:
        for label in labels:
            if scores[label]:
                logger.info(
                    f"[{backend}] {label:9s} state accuracy "
                    f"{correct[label] / len(scores[label]):.2f}"
                )
        violin(
            scores,
            out_dir / f"state-score-{backend}.png",
        )
    elif not args.no_states:
        logger.warning(
            f"[{backend}] the cache holds no state scores; pass --resample to "
            "re-measure with states"
        )


def measure(backend: str, args, out_dir: Path, cache: Path) -> dict:
    """Run the measurement for one keypoint backend and cache it."""
    select_encoder(args.scene, args.backbone, args.stride, backend)
    scene = Scene.get(scene_config(args.scene))
    model = ExpertModel.get(experts_for_scene(args.scene)[0], auto_load=False)
    model.use_gt(False)
    model.use_states = not args.no_states
    model.load()
    labels = list(scene.entities)

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
        "backend": backend,
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
    return payload


if __name__ == "__main__":
    main()
