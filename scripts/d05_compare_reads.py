"""Compare the visual read of demo frames with the visual read of live frames.

Both sides use the same code and the same references; nothing here touches ground
truth. The conditions are fitted on the demo-frame reads (`from_image(demo[0])`),
the arm reads live frames, so those two distributions have to agree -- when they
do not, every gate rejects and the run discards every episode while looking
healthy.

    python -m scripts.d05_compare_reads --scene scene0 --fit-frames 6 --act-frames 3
"""

import argparse

import h5py
import numpy as np

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


def read(reader, frame):
    reader.reset_tracking()
    poses = reader.kp_extractor.extract_poses(frame)
    return {label: kp.xyz for label, kp in poses.items()}, {
        label: kp.score for label, kp in poses.items()
    }


def depth_range(frame) -> tuple[float, float]:
    depth = np.asarray(frame.d)
    return float(depth.min()), float(depth.max())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    add_threads_argument(parser)
    parser.add_argument("--fit-frames", type=int, default=6, help="Demo frames.")
    parser.add_argument("--act-frames", type=int, default=3, help="Live frames.")
    args = parser.parse_args()
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    select_encoder(args.scene, args.backbone, args.stride, args.kp_backend)
    scene = Scene.get(scene_config(args.scene))
    cfg = experts_for_scene(args.scene)[0]
    model = ExpertModel.get(cfg, auto_load=False)
    model.use_gt(False)
    model.use_states = False  # positions only; the state read is a separate gate
    model.load()
    labels = list(scene.entities)
    demos_path = model.load_dir(cfg) / "demos.h5"

    with h5py.File(demos_path, "r") as demos:
        total = len(demos["rgb"])
        rows = np.unique(
            np.linspace(0, total - 1, max(2, args.fit_frames)).astype(int)
        )
        fit, fit_scores, fit_depth = [], [], []
        for row in rows:
            image = {
                key: demos[key][row]
                for key in IMAGE_KEYS
                if key != "demo"
            }
            extra = {
                key: demos[key][row]
                for key in demos.keys()
                if key not in IMAGE_KEYS
            }
            obs, _ = scene.to_internal(image, extra)
            frame = scene.to_td_image(obs)
            positions, scores = read(model, frame)
            fit.append(positions)
            fit_scores.append(scores)
            fit_depth.append(depth_range(frame))

    act, act_scores, act_depth = [], [], []
    for _ in range(max(1, args.act_frames)):
        (_, x_frame), _ = scene.sample_task(with_scenes=False)
        positions, scores = read(model, x_frame)
        act.append(positions)
        act_scores.append(scores)
        act_depth.append(depth_range(x_frame))

    fd, ad = np.asarray(fit_depth), np.asarray(act_depth)
    logger.info(f"demo frames: {len(rows)} of {total}, depth {fd.mean(0).round(3)}")
    logger.info(f"live frames: {len(act)}, depth {ad.mean(0).round(3)}")
    worst = 0.0
    for label in labels:
        fitted = np.stack([row[label] for row in fit])
        live = np.stack([row[label] for row in act])
        shift = float(np.linalg.norm(live.mean(0) - fitted.mean(0)))
        worst = max(worst, shift)
        logger.info(
            f"  {label:8s} demo mean {np.round(fitted.mean(0), 3)} "
            f"std {np.round(fitted.std(0), 3)} score "
            f"{np.mean([row[label] for row in fit_scores]):.2f} | "
            f"live mean {np.round(live.mean(0), 3)} std {np.round(live.std(0), 3)} "
            f"score {np.mean([row[label] for row in act_scores]):.2f} | "
            f"shift {shift:.2f} m"
        )
    logger.info(f"=== largest mean shift {worst:.2f} m")
    raise SystemExit(1 if worst > 0.1 else 0)


if __name__ == "__main__":
    main()
