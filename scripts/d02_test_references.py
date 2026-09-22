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
    # judge every frame on its own: the episode prior would smooth over exactly the
    # reference errors this script exists to find
    model.kp_extractor.cfg.tracking = TrackingMode.NONE
    model.kp_extractor.reset_tracking()

    dataset = h5py.File(demo_dataset(scene_cfg, args.dataset), "r")
    frames = min(args.frames, len(dataset["rgb"]))

    position: dict[str, list[float]] = {l: [] for l in scene.entities}
    joint: dict[str, list[float]] = {l: [] for l in scene.entities}
    states: dict[str, list[bool]] = {l: [] for l in scene.entities}

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
                **{
                    name: scene.references[label][name].xyz
                    for name in entity.reference_names
                },
            }
            wanted = entity.extra_part(label, truth)
            if len(wanted):
                got = entity.extra_from_references(label, positions)
                joint[label].append(float(np.max(np.abs(got - wanted))))
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
    print(
        "  (position is compared to the recorded pose of the same frame; the joint\n"
        "   value is compared in its own units: a slide fraction, or sin/cos of an\n"
        "   angle - so a mean well under 0.1 means the reference geometry is right)"
    )


if __name__ == "__main__":
    main()
