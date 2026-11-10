import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import h5py  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib import patheffects as pe  # noqa: E402
from matplotlib.patches import Circle
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
    annotation = axis.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(8, 8),
        color=color,
        fontsize=9,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    annotation.set_in_layout(False)  # the label must not resize the panel


def mark_dashed(
    axis,
    x: float,
    y: float,
    color: str,
    text: str,
    dashes: tuple[int, int, int] | None = None,
    marker_pt: float = 10.0,
    text_offset: int = 8,
) -> None:

    bbox = axis.get_window_extent()
    xlo, xhi = axis.get_xlim()
    px_per_data = bbox.width / (xhi - xlo) if xhi > xlo else 1.0
    radius = (marker_pt * axis.figure.dpi / 72.0) / 2.0 / px_per_data
    if dashes is None:
        linestyle = (0, ())
    else:  # (offset, on, off)
        linestyle = (dashes[0], (dashes[1], dashes[2]))
    axis.add_patch(
        Circle(
            (x, y),
            radius=radius,
            fill=False,
            edgecolor=color,
            linewidth=2,
            linestyle=linestyle,
        )
    )
    annotation = axis.annotate(
        text,
        (x, y),
        textcoords="offset points",
        xytext=(text_offset, 8),
        color=color,
        fontsize=9,
        fontweight="bold",
        path_effects=[pe.withStroke(linewidth=1.2, foreground="black")],
    )
    annotation.set_in_layout(False)  # the label must not resize the panel


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


def flow_color(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Dense-correspondence colour wheel: hue = direction, saturation = magnitude."""
    from matplotlib.colors import hsv_to_rgb

    magnitude = np.hypot(dx, dy)
    peak = float(magnitude.max()) if magnitude.size else 0.0
    saturation = (
        np.clip(magnitude / peak, 0.0, 1.0) if peak > 0 else np.zeros_like(magnitude)
    )
    angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)  # [0, 1]
    return hsv_to_rgb(np.stack([angle, saturation, np.ones_like(angle)], axis=-1))


def colour_wheel(radius_px: int = 48) -> np.ndarray:
    """An RGB colour-wheel legend: hue = angle, saturation = radius, white outside."""
    from matplotlib.colors import hsv_to_rgb

    grid = np.linspace(-1.0, 1.0, 2 * radius_px + 1)
    xx, yy = np.meshgrid(grid, grid)
    radius = np.hypot(xx, yy)
    hue = (np.arctan2(yy, xx) + np.pi) / (2 * np.pi)
    wheel = hsv_to_rgb(
        np.stack([hue, np.clip(radius, 0.0, 1.0), np.ones_like(radius)], axis=-1)
    )
    wheel[radius > 1.0] = 1.0
    return wheel


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

    dino_cfg = select_encoder(args.scene, args.backbone, args.stride, "dino")
    roma_cfg = select_encoder(args.scene, "", 0, "roma")
    model = ExpertModel.get(find_scene_models(args.scene)[0], auto_load=False)
    dino = ImageEncoder.get(dino_cfg)
    dino.prepare_for_scene(scene_cfg)
    # each frame is located on its own: a belief carried in from another frame
    # would mix a tracking error with the reference error this is here to show
    dino.cfg.tracking = TrackingMode.NONE
    dino.reset_tracking()
    roma = ImageEncoder.get(roma_cfg)
    roma.prepare_for_scene(scene_cfg)
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
        maps, confidence = dino.similarity_maps(image)
        dino_poses = dino.extract_poses(image)
        roma_poses = roma.extract_poses(image)
        rgb = (image.rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
        height, width = rgb.shape[:2]

        for label in wanted:
            reference: Reference = scene.references[label][Reference.POSITION]
            clicked = (reference.x, reference.y)
            dino_reported = (dino_poses[label].x, dino_poses[label].y)
            roma_reported = (roma_poses[label].x, roma_poses[label].y)
            gap = float(
                np.hypot(dino_reported[0] - clicked[0], dino_reported[1] - clicked[1])
            )
            moved[label].append(gap)

            # dense-correspondence panel (dino)
            similarity = np.asarray(maps[dino.kp_labels.index(label)].detach().cpu())
            grid_h, grid_w = similarity.shape
            stride, patch = dino.cfg.stride, dino.patch_size
            # a map on the token grid has to be placed at its window centres; an
            # upsampled map is already at the frame size, so an index is a pixel
            on_lattice = (grid_h, grid_w) == dino.compute_patch_grid_size(
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
            else:
                extent = (-0.5, width - 0.5, height - 0.5, -0.5)

            # warp panel (roma)
            ref_image, _, _ = roma.references[label]
            frame_rgb = image.rgb
            if frame_rgb.ndim == 3:
                frame_rgb = frame_rgb[None]
            preds = roma.model.match(ref_image, frame_rgb.to(roma.device))
            warp = preds["warp_AB"]
            if warp.ndim == 4:
                warp = warp[0]
            warp = warp.permute(2, 0, 1).cpu().numpy()  # (2, H, W) normalised
            gh, gw = warp.shape[1], warp.shape[2]
            warp_ys = np.linspace(-1.0, 1.0, gh)
            warp_xs = np.linspace(-1.0, 1.0, gw)
            warp_gx, warp_gy = np.meshgrid(warp_xs, warp_ys)
            warp_rgb = flow_color(warp[0] - warp_gx, warp[1] - warp_gy)

            def draw(ax_ref, ax_frame, ax_dc, ax_warp) -> None:
                ax_ref.imshow(
                    np.asarray(reference.image), extent=(0, width, height, 0)
                )
                mark_dashed(
                    ax_ref,
                    clicked[0],
                    clicked[1],
                    "springgreen",
                    "reference",
                )

                ax_frame.imshow(rgb)
                mark_dashed(
                    ax_frame,
                    dino_reported[0],
                    dino_reported[1],
                    "deeppink",
                    "dino",
                    dashes=(0, 2, 2),
                )
                mark_dashed(
                    ax_frame,
                    roma_reported[0],
                    roma_reported[1],
                    "cyan",
                    "roma",
                    dashes=(2, 2, 2),
                    text_offset=-32,
                )

                ax_dc.imshow(similarity, cmap="viridis", extent=extent)
                dc_legend = ax_dc.inset_axes([0.02, 0.02, 0.10, 0.10])
                dc_legend.imshow(
                    np.linspace(0.0, 1.0, 64)[None, :], cmap="viridis", aspect="auto"
                )
                dc_legend.set_xticks([])
                dc_legend.set_yticks([])
                for spine in dc_legend.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.6)

                ax_warp.imshow(warp_rgb, extent=(0, width, height, 0))
                warp_legend = ax_warp.inset_axes([0.02, 0.02, 0.10, 0.10])
                warp_legend.imshow(colour_wheel(), aspect="equal")
                warp_legend.set_xticks([])
                warp_legend.set_yticks([])
                for spine in warp_legend.spines.values():
                    spine.set_edgecolor("black")
                    spine.set_linewidth(0.6)

                for ax in (ax_ref, ax_frame, ax_dc, ax_warp):
                    ax.set_xticks([])
                    ax.set_yticks([])

            # 1 x 4 strip: reference, frame, dense correspondence, warp
            figure, axes = plt.subplots(1, 4, figsize=(18.0, 5.0))
            draw(*axes)
            figure.tight_layout()
            figure.savefig(
                out_dir / f"predictions_strip_{frame:06d}_{label}.png",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(figure)

            # 2 x 2 grid: reference + frame on top, dense correspondence + warp below
            figure, axes = plt.subplots(2, 2, figsize=(9.0, 9.0))
            draw(axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1])
            figure.tight_layout(h_pad=1.1, w_pad=1.1)
            figure.savefig(
                out_dir / f"predictions_grid_{frame:06d}_{label}.png",
                dpi=120,
                bbox_inches="tight",
            )
            plt.close(figure)
            print(
                f"  frame {frame:>7} {label:<9} clicked {clicked} "
                f"dino {dino_reported} roma {roma_reported} "
                f"-> {gap:5.1f} px apart (dino)"
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
