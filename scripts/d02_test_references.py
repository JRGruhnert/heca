import argparse

import h5py
import numpy as np

from heca.data.reference import Reference
from heca.experts.expert import ExpertModel
from heca.guis.scene_sample_selector import demo_dataset
from heca.image_encoders.dino_encoder import TrackingMode
from heca.image_encoders.image_encoder import ImageEncoder
from heca.scenes.scene import Scene

from scripts.common.args import add_encoder_arguments, add_scene_argument
from scripts.common.scenes import (
    find_scene_config,
    find_scene_models,
    select_encoder,
)


def frame_image(scene: Scene, dataset, frame: int):
    """One dataset frame as the encoder's ``TDImage``."""
    depth = dataset["depth"][frame]
    return scene.to_td_image(
        {
            "image": {
                "rgb": dataset["rgb"][frame],
                "depth": depth,
                "mask": np.zeros_like(depth, dtype=np.uint8),
                "extrinsics": dataset["extrinsics"][frame],
                "intrinsics": dataset["intrinsics"][frame],
            }
        }
    )


def offset_report(
    offsets: dict[int, np.ndarray],
    correction: np.ndarray | None,
    moved: dict[int, float] | None = None,
) -> dict:
    """The correction, the residual it leaves, and how that residual tracks travel.

    ``offsets`` maps a frame to ``read - recorded``. ``correction`` is the offset
    the click carries; ``None`` falls back to the median offset, for a reference
    that does not record its frame.
    """
    if not offsets:
        return {
            "offset": 0.0,
            "correction": np.zeros(3),
            "min": 0.0,
            "mean": 0.0,
            "max": 0.0,
            "travel": 0.0,
            "slope": float("nan"),
        }

    frames = sorted(offsets)
    vectors = np.stack([offsets[f] for f in frames])
    if correction is None:
        correction = np.median(vectors, axis=0)
    residual = np.linalg.norm(vectors - np.asarray(correction, dtype=float), axis=1)

    travel, slope = 0.0, float("nan")
    if moved:
        distance = np.array([moved[f] for f in frames], dtype=float)
        travel = float(distance.max())
        if travel > 0.0 and distance.std() > 0.0 and residual.std() > 0.0:
            # residual against how far the object moved: for frames that differ by a
            # rotation this grows linearly, with a slope near the rotation angle
            slope = float(np.polyfit(distance, residual, 1)[0])

    return {
        "offset": float(np.linalg.norm(vectors, axis=1).mean()),
        "correction": np.asarray(correction, dtype=float),
        "min": float(residual.min()),
        "mean": float(residual.mean()),
        "max": float(residual.max()),
        "travel": travel,
        "slope": slope,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    parser.add_argument("--frames", type=int, default=10, help="frames to check")
    parser.add_argument(
        "--states",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also read the categorical states (loads Molmo); --no-states skips it",
    )
    parser.add_argument(
        "--dataset",
        default="",
        help="demo .h5 to check against; default: the scene's own",
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
    # judge every frame on its own: the episode prior would smooth over exactly the
    # reference errors this script exists to find
    keypoints.cfg.tracking = TrackingMode.NONE
    keypoints.reset_tracking()
    states_encoder = None
    if args.states:
        states_encoder = ImageEncoder.get(model.cfg.state_extraction)
        states_encoder.prepare_for_scene(scene_cfg)

    scene = model.scene
    dataset = h5py.File(demo_dataset(scene_cfg, args.dataset), "r")
    constant = min(args.frames, len(dataset["rgb"]))

    # the joint value is compared in the units the scene normalizes with, so the
    # travel the demos recorded is read off the same file the frames come from
    for label, entity in scene.entities.items():
        values = entity.extra_values(label, dataset)
        if values is not None:
            scene.add_extra_range(label, *values)

    # the click's own offset comes from the click, encoder-free; the frame it was
    # clicked on is added to the frames read, so the residual is measured from the
    # same starting point
    anchor_frames: dict[str, int | None] = {}
    corrections: dict[str, np.ndarray | None] = {}
    for label in scene.entities:
        reference = scene.references.get(label, {}).get(Reference.POSITION)
        if reference is None:
            continue
        if reference.frame is None:
            anchor_frames[label] = None
            corrections[label] = None
            continue
        frame = int(reference.frame)
        anchor_frames[label] = frame
        recorded = np.asarray(dataset[f"heca_{label}_pos"][frame], dtype=float)
        corrections[label] = np.asarray(reference.xyz, dtype=float) - recorded

    frames = sorted(
        {*range(constant), *[f for f in anchor_frames.values() if f is not None]}
    )

    offsets: dict[str, dict[int, np.ndarray]] = {l: {} for l in scene.entities}
    recorded_at: dict[str, dict[int, np.ndarray]] = {l: {} for l in scene.entities}
    joint: dict[str, list[float]] = {l: [] for l in scene.entities}
    states: dict[str, list[bool]] = {l: [] for l in scene.entities}
    skipped: list[str] = []

    for frame in frames:
        image = frame_image(scene, dataset, frame)
        poses = keypoints.extract_poses(image)
        read_states = states_encoder.extract_states(image) if states_encoder else {}
        truth = {
            key: dataset[key][frame]  # type: ignore
            for key in dataset.keys()  # type: ignore
            if key.startswith("heca_")
        }
        for label, entity in scene.entities.items():
            recorded = np.asarray(truth[f"heca_{label}_pos"], dtype=float)
            read = np.asarray(poses[label].xyz, dtype=float)
            offsets[label][frame] = read - recorded
            recorded_at[label][frame] = recorded

            positions = {
                "current": poses[label].xyz,
                "reference": scene.references[label][Reference.POSITION].xyz,
                **scene.extra_references.get(label, {}),
                **{
                    name: scene.references[label][name].xyz
                    for name in entity.reference_names
                },
            }
            travel = scene.extra_ranges.get(label)
            if len(entity.reference_names) or (
                label in scene.extra_references and travel is not None
            ):
                extra_range = travel if label in scene.extra_references else None
                wanted = entity.extra_part(label, truth, extra_range=extra_range)
                if len(wanted):
                    got = entity.extra_from_references(label, positions)
                    joint[label].append(float(np.max(np.abs(got - wanted))))
            elif label not in skipped:
                skipped.append(label)

            if label in read_states:
                states[label].append(
                    read_states[label][0] == int(truth[f"heca_{label}_ste"][0])
                )

    print(
        f"{args.scene}: {len(frames)} frames of {demo_dataset(scene_cfg, args.dataset)}"
    )
    print(
        f"  {'label':<9} {'correction (mm)':>22} {'residual mm':>18}"
        f" {'moved':>8} {'slope':>6} | {'joint':>7} | {'state':>6}"
    )
    for label in scene.entities:
        frame = anchor_frames.get(label)
        moved: dict[int, float] = {}
        if frame is not None:
            start = recorded_at[label][frame]
            moved = {
                f: float(np.linalg.norm(point - start))
                for f, point in recorded_at[label].items()
            }
        report = offset_report(offsets[label], corrections.get(label), moved)
        correction = ", ".join(f"{1000 * v:+.0f}" for v in report["correction"])
        residual = (
            f"{1000 * report['min']:5.0f}/{1000 * report['mean']:5.0f}/"
            f"{1000 * report['max']:5.0f}"
        )
        line = (
            f"  {label:<9} {correction:>22} {residual:>18} "
            f"{1000 * report['travel']:6.0f}mm {report['slope']:6.2f}"
        )
        if joint[label]:
            line += f" | {np.mean(joint[label]):7.3f}"
        else:
            line += f" | {'-':>7}"
        if states[label]:
            line += f" | {100 * float(np.mean(states[label])):5.0f}%"
        else:
            line += f" | {'-':>6}"
        print(line)

    anchored = [label for label, frame in anchor_frames.items() if frame is not None]
    print(
        "  correction = the offset the click carries (clicked point against the"
        "\n               recording on the frame it was clicked on), per axis in mm:"
        "\n               the value to apply to the vision side before comparing"
        "\n               ("
        + (
            f"{len(anchored)} of {len(anchor_frames)} references record their frame"
            if anchored
            else "no reference records its frame: median offset used instead"
        )
        + ")"
        "\n  residual   = smallest/mean/largest of what is left per frame once the"
        "\n               correction is applied. This is the tracking error"
        "\n  moved      = how far the object travelled from its pose on the click frame"
        "\n  slope      = residual per mm moved. Near 0: one constant correction is"
        "\n               enough. Clearly positive: the frames differ by a rotation,"
        "\n               which no single offset can absorb"
        "\n  joint      = mean |vision fraction - recorded fraction|; well under 0.1"
        "\n  state      = share of frames the categorical readout agrees with the recording"
        "\n  (the correction is per label: each entity's recorded position is its own site)"
    )
    if skipped:
        print(
            f"  joint value not compared for {', '.join(skipped)}: a slide's ends are\n"
            "   derived from its demos, and no run has derived them into this scene yet"
        )
    dataset.close()
    scene.close()


if __name__ == "__main__":
    main()
