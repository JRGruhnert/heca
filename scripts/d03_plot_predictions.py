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
from scripts.common.args import add_scene_argument  # noqa: E402
from scripts.common.scenes import find_scene_config, find_scene_models  # noqa: E402
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
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
            axes[1].set_title(
                f"frame {frame}: what the encoder reports ({reported[0]}, {reported[1]}), "
                f"{gap:.0f} px away",
                fontsize=10,
            )
            mark(axes[1], reported[0], reported[1], "red", "o", "reported")

            similarity = np.asarray(
                maps[keypoints.kp_labels.index(label)].detach().cpu()
            )
            axes[2].imshow(
                similarity,
                cmap="viridis",
                extent=(-0.5, width - 0.5, height - 0.5, -0.5),
            )
            axes[2].set_title(
                f"similarity map, peak {float(confidence[keypoints.kp_labels.index(label)]):.2f}",
                fontsize=10,
            )
            mark(
                axes[2],
                reported[0] / (width - 1) * (similarity.shape[1] - 1),
                reported[1] / (height - 1) * (similarity.shape[0] - 1),
                "red",
                "o",
                "",
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
