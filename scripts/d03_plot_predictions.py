import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from heca.data.reference import Reference  # noqa: E402
from heca.experts.expert import ExpertModel  # noqa: E402
from heca.guis.scene_sample_selector import demo_dataset  # noqa: E402
from heca.image_encoders.dino_encoder import TrackingMode  # noqa: E402
from heca.image_encoders.image_encoder import ImageEncoder  # noqa: E402
from heca.scenes.scene import Scene  # noqa: E402
from scripts.common.args import (  # noqa: E402
    add_encoder_arguments,
    add_scene_argument,
)
from scripts.common.scenes import (  # noqa: E402
    find_scene_config,
    find_scene_models,
    select_encoder,
)
from scripts.d02_test_references import frame_image  # noqa: E402


def mark(axis, x: float, y: float, color: str, marker: str, text: str) -> None:
    axis.plot(
        x,
        y,
        marker,
        color=color,
        markersize=13,
        markeredgewidth=2,
        markerfacecolor="none",
        linestyle="none",
    )
    axis.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(8, 8),
        color=color,
        fontsize=9,
        fontweight="bold",
    )


def lattice(
    grid_hw: tuple[int, int], patch: int, stride: int
) -> tuple[list[float], list[float]]:
    """Where each token's window sits in frame pixels: kernel ``patch`` slid by ``stride``.

    Token ``(row, column)`` is the patch centred on ``stride * i + patch / 2``, so at
    stride 8 with 16 px patches the 31 centres run 8, 16, ... 248 -- not 0 ... 255.
    """
    rows = [r * stride + patch / 2 for r in range(grid_hw[0])]
    columns = [c * stride + patch / 2 for c in range(grid_hw[1])]
    return rows, columns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    parser.add_argument("--frames", type=int, default=3, help="frames to draw")
    parser.add_argument(
        "--spread",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="spread the frames over the dataset instead of taking its first ones",
    )
    parser.add_argument(
        "--dataset", default="", help="dataset to draw from; default: the scene's own"
    )
    parser.add_argument(
        "--labels", default="", help="comma separated labels; default: all of them"
    )
    parser.add_argument(
        "--out",
        default="",
        help="where to write the PNGs; default: <scene>/plots/predictions",
    )
    parser.add_argument(
        "--grid",
        action="store_true",
        help="draw the descriptor lattice: one dot per token on the frame, and the "
        "map aligned to those dots instead of stretched over the whole frame",
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    if scene_cfg is None:
        parser.error(f"No scene found with tag {args.scene!r}")

    scene = Scene.get(scene_cfg)
    if not scene.references_ready():
        parser.error(
            f"{args.scene}: no complete reference set in "
            f"{Scene.save_dir(scene_cfg)}; run scripts/d01_select_references.py"
        )

    select_encoder(args.scene, args.backbone, args.stride)
    model = ExpertModel.get(find_scene_models(args.scene)[0], auto_load=False)
    keypoints = ImageEncoder.get(model.cfg.kp_extraction)
    keypoints.prepare_for_scene(scene_cfg)
    # each frame is located on its own: a belief carried in from another frame
    # would mix a tracking error with the reference error this is here to show
    keypoints.cfg.tracking = TrackingMode.NONE
    keypoints.reset_tracking()
    scene = model.scene

    dataset_path = demo_dataset(scene_cfg, args.dataset)
    dataset = h5py.File(dataset_path, "r")
    total = len(dataset["rgb"])
    wanted = [w.strip() for w in args.labels.split(",") if w.strip()] or list(
        scene.entities
    )
    wanted = [w for w in wanted if w in scene.references]
    indices = (
        np.linspace(0, total - 1, args.frames).astype(int)
        if args.spread
        else np.arange(min(args.frames, total))
    )

    out_dir = (
        Path(args.out)
        if args.out
        else Scene.save_dir(scene_cfg) / "plots" / "predictions"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{args.scene}: {len(indices)} frame(s) of {dataset_path}")
    print("  (all numbers are pixels in the 256x256 frame; nothing is projected)")

    moved: dict[str, list[float]] = {w: [] for w in wanted}
    for frame in indices:
        image = frame_image(scene, dataset, frame)
        maps, confidence = keypoints.similarity_maps(image)
        poses = keypoints.extract_poses(image)
        rgb = (image.rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        height, width = rgb.shape[:2]

        for label in wanted:
            reference: Reference = scene.references[label][Reference.POSITION]
            clicked = (reference.x, reference.y)
            reported = (poses[label].x, poses[label].y)
            gap = float(np.hypot(reported[0] - clicked[0], reported[1] - clicked[1]))
            moved[label].append(gap)

            figure, axes = plt.subplots(1, 3, figsize=(13.5, 5.0))
            axes[0].imshow(np.asarray(reference.image))
            axes[0].set_title(
                f"{label}: what you clicked ({clicked[0]}, {clicked[1]})", fontsize=10
            )
            mark(axes[0], clicked[0], clicked[1], "red", "o", "clicked")

            axes[1].imshow(rgb)

            similarity = np.asarray(
                maps[keypoints.kp_labels.index(label)].detach().cpu()
            )
            grid_h, grid_w = similarity.shape
            stride, patch = keypoints.cfg.stride, keypoints.patch_size
            # a map on the token grid has to be placed at its window centres; an
            # upsampled map is already at the frame size, so an index is a pixel
            on_lattice = (grid_h, grid_w) == keypoints.compute_patch_grid_size(
                (height, width)
            ) and grid_h < height
            if on_lattice:
                rows, columns = lattice((grid_h, grid_w), patch, stride)
            else:
                rows = [float(r) for r in range(grid_h)]
                columns = [float(c) for c in range(grid_w)]
            best = np.unravel_index(int(similarity.argmax()), similarity.shape)
            peak_x, peak_y = columns[best[1]], rows[best[0]]
            if args.grid and on_lattice:
                # span the window centres really span, so a cell sits over the pixels
                # its descriptor was read from instead of being stretched to the frame
                half = stride / 2
                extent = (
                    columns[0] - half,
                    columns[-1] + half,
                    rows[-1] + half,
                    rows[0] - half,
                )
                axes[1].plot(
                    np.tile(columns, len(rows)),
                    np.repeat(rows, len(columns)),
                    ".",
                    color="orange",
                    markersize=1.5,
                )
            else:
                extent = (-0.5, width - 0.5, height - 0.5, -0.5)
            shape_text = (
                f"{grid_h}x{grid_w} tokens, {patch}px window every {stride}px"
                if on_lattice
                else f"{grid_h}x{grid_w} px, descriptors upsampled to the frame"
            )
            axes[2].imshow(similarity, cmap="viridis", extent=extent)
            axes[2].set_title(
                f"similarity map {shape_text}, "
                f"peak {float(confidence[keypoints.kp_labels.index(label)]):.2f}",
                fontsize=10,
            )
            mark(axes[2], peak_x, peak_y, "red", "o", "")
            if args.grid and on_lattice:
                # the token's own centre, next to where the encoder says it is
                mark(axes[1], peak_x, peak_y, "orange", "x", "token")
            mark(axes[1], reported[0], reported[1], "red", "o", "reported")
            axes[1].set_title(
                f"frame {frame}: reports ({reported[0]}, {reported[1]}), "
                f"{gap:.0f} px from the click",
                fontsize=10,
            )
            axes[2].set_yticklabels([])
            axes[2].set_xticklabels([])
            print(
                f"  {'':>4} {label:<9} map {shape_text.split(',')[0]}, peak "
                f"({peak_x:.0f}, {peak_y:.0f})"
            )

            for axis in axes:
                axis.set_xticks([])
                axis.set_yticks([])
            figure.tight_layout()
            path = out_dir / f"predictions_{frame:06d}_{label}.png"
            figure.savefig(path, dpi=120, bbox_inches="tight")
            plt.close(figure)
            print(
                f"  frame {frame:>7} {label:<9} clicked {clicked} reported {reported} "
                f"-> {gap:5.1f} px apart, confidence {poses[label].score:.2f}"
            )

    print("  summary, pixels between the clicked and the reported point per label:")
    for label, gaps in moved.items():
        errors = np.asarray(gaps)
        print(
            f"    {label:<9} mean {errors.mean():6.1f} px, worst {errors.max():6.1f} px"
        )
    print(f"  wrote to {out_dir}")
    dataset.close()
    scene.close()


if __name__ == "__main__":
    main()
