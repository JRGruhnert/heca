import argparse

import h5py
import numpy as np

from heca.experts.expert import ExpertModel
from heca.image_encoders.dino_encoder import TrackingMode
from heca.guis.scene_sample_selector import demo_dataset
from heca.scenes.scene import Scene

from scripts.common.args import add_scene_argument
from scripts.common.scenes import find_scene_config, find_scene_models


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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    parser.add_argument("--frames", type=int, default=10, help="frames to check")
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

    model = ExpertModel.get(find_scene_models(args.scene)[0], auto_load=False)
    model.use_gt(False)  # loads the scene and builds the encoders from references
    model.kp_extractor.cfg.tracking = TrackingMode.NONE
    model.kp_extractor.reset_tracking()

    scene = model.scene
    dataset = h5py.File(demo_dataset(scene_cfg, args.dataset), "r")
    frames = min(args.frames, len(dataset["rgb"]))

    for label, entity in scene.entities.items():
        values = entity.extra_values(label, dataset)
        if values is not None:
            scene.add_extra_range(label, *values)

    position: dict[str, list[float]] = {l: [] for l in scene.entities}
    joint: dict[str, list[float]] = {l: [] for l in scene.entities}
    states: dict[str, list[bool]] = {l: [] for l in scene.entities}
    skipped: list[str] = []

    for frame in range(frames):
        image = frame_image(scene, dataset, frame)
        poses = model.kp_extractor.extract_poses(image)
        read_states = model.ste_extractor.extract_states(image)
        truth = {
            key: dataset[key][frame]
            for key in dataset.keys()  # type: ignore
            if key.startswith("heca_")
        }
        for label, entity in scene.entities.items():
            position[label].append(
                float(np.linalg.norm(poses[label].xyz - truth[f"heca_{label}_pos"]))
            )
            positions = {
                "current": poses[label].xyz,
                "reference": scene.references[label]["pos"].xyz,
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

    print(f"{args.scene}: {frames} frames of {demo_dataset(scene_cfg, args.dataset)}")
    for label in scene.entities:
        errors = np.asarray(position[label])
        line = (
            f"  {label:<9} position {1000 * errors.mean():7.1f} mm mean, "
            f"{1000 * errors.max():7.1f} mm worst"
        )
        if joint[label]:
            joint_errors = np.asarray(joint[label])
            line += (
                f" | joint value {joint_errors.mean():.3f} mean, "
                f"{joint_errors.max():.3f} worst"
            )
        if states[label]:
            line += f" | state {100 * float(np.mean(states[label])):.0f}% right"
        print(line)
    if skipped:
        print(
            f"  joint value not compared for {', '.join(skipped)}: a slide's ends are\n"
            "   derived from its demos, and no run has derived them into this scene\n"
            "   yet (fit the variant, or run a visual pass, then repeat this)"
        )
    print(
        "  (position is compared to the recorded pose of the same frame; the joint\n"
        "   value is compared in its own units: a slide fraction, or sin/cos of an\n"
        "   angle - so a mean well under 0.1 means the reference geometry is right)"
    )


if __name__ == "__main__":
    main()
