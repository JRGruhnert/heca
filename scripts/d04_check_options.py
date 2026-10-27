"""Ask sampled episodes whether any option applies, and why not.

The training loop drops an episode whose start state no option accepts. That is
cheap and harmless, but it produces no "Predicted ... steps" line, so a condition
set that rejects every start state looks like a hang. This runs the same
sample -> encode -> graph.build path without training, and prints the gate that
rejected, so a broken fit is visible before a run is started.

    python -m scripts.d04_check_options --scene scene0 --episodes 10
    python -m scripts.d04_check_options --scene scene0 --episodes 10 --gt
"""

import argparse
import faulthandler

import numpy as np

from heca.data.data import DCScene
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.misc import logger
from heca.scenes.scene import Scene
from scripts.common.args import (
    add_encoder_arguments,
    add_scene_argument,
    add_threads_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import experts_for_scene, scene_config, select_encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    add_use_gt_argument(parser)
    add_threads_argument(parser)
    parser.add_argument(
        "--episodes", type=int, default=10, help="Start states to sample."
    )
    parser.add_argument(
        "--stack-dump",
        type=int,
        default=0,
        metavar="SECONDS",
        help="Dump every thread's stack every SECONDS (0 = off). Needs no ptrace, "
        "so it works where py-spy is refused.",
    )
    parser.add_argument(
        "--explain",
        action="store_true",
        help="Print the read value, the fitted mean and the residual per entity "
        "for the first episode, so a distance rejection is in metres, not sigma.",
    )
    parser.add_argument(
        "--dump-frames",
        default="",
        metavar="DIR",
        help="Save the start and goal frame of the first episode to DIR.",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="*",
        default=None,
        help="Fixed seeds for reproducible sampling (default: random).",
    )
    args = parser.parse_args()
    if args.stack_dump:
        # a hang is silent otherwise: the loop that hangs prints nothing, and the
        # log simply stops
        faulthandler.dump_traceback_later(args.stack_dump, repeat=True)
    if args.threads:
        import torch

        torch.set_num_threads(args.threads)

    select_encoder(args.scene, args.backbone, args.stride, args.kp_backend)
    cfgs = experts_for_scene(args.scene)
    scene: Scene = Scene.get(scene_config(args.scene))

    experts = []
    for cfg in cfgs:
        # same order as Heca: the encoders are built before the policy is read, so
        # a visual check does not load the ground-truth policy first
        model = ExpertModel.get(cfg, auto_load=False)
        model.use_gt(args.gt)
        model.load()
        experts.append(model)
    scene.finish_extras()

    graph = Graph.generate(
        cfgs,
        SubgoalMode.BOTH,
        use_rotation=False,
        position_jitter=0.0,
    )
    if not args.gt:
        import torch

        if torch.cuda.is_available():
            free = [
                torch.cuda.mem_get_info(index)[0] / 2**30
                for index in range(torch.cuda.device_count())
            ]
            listed = ", ".join(f"{i}:{gib:.1f} GiB" for i, gib in enumerate(free))
            logger.info(f"free per visible card -- {listed}")
            if sum(free) < 10.0:
                logger.error(
                    "Molmo alone is 9.3 GiB of weights, so this needs >= 10 GiB free "
                    "across the visible cards; they are held by another process. "
                    "Kill it, or point CUDA_VISIBLE_DEVICES at idle cards."
                )
                raise SystemExit(2)

    reader = experts[0]
    logger.info(
        f"{len(cfgs)} experts, use_gt={args.gt}, "
        f"{len(scene.entities)} entities, {len(graph.ns_option.items)} options"
    )

    rejected = 0
    reasons: dict[str, int] = {}
    for episode in range(args.episodes):
        if args.seeds:
            import torch

            seed = args.seeds[episode % len(args.seeds)]
            torch.manual_seed(seed)
            np.random.seed(seed)

        (x, x_frame), (y, y_frame) = scene.sample_task(with_scenes=args.gt)
        for model in experts:
            model.reset_tracking()
        start = reader.make_scene(x, x_frame)
        reader.reset_tracking()
        goal = reader.make_scene(y, y_frame)
        if args.gt:
            start, goal = x, y

        if episode == 0 and (args.explain or args.dump_frames):
            if args.dump_frames:
                import os

                from PIL import Image

                os.makedirs(args.dump_frames, exist_ok=True)
                for name, frame in (("start", x_frame), ("goal", y_frame)):
                    rgb = frame.rgb
                    if hasattr(rgb, "detach"):
                        rgb = rgb.detach().cpu().permute(1, 2, 0).numpy()
                    array = (rgb * 255.0).astype("uint8") if rgb.max() <= 1.0 else rgb
                    Image.fromarray(array).save(f"{args.dump_frames}/{name}.png")
                    logger.info(f"saved {args.dump_frames}/{name}.png {array.shape}")
            if args.explain:
                pair = reader.conditions

                def compare(role, condition, scene_read):
                    logger.info(f"  -- {role} ({condition.label})")
                    for label, entity in scene.entities.items():
                        read = np.asarray(scene_read[label].value, dtype=float)
                        # exactly what Condition.test scores: the variance floor and
                        # the component choice both come from here
                        up = condition.models[label].get_parameters().copy()
                        p = entity.scoring_parameters(up)
                        sample = entity.model_value(read)
                        groups = entity.pose_groups(True)
                        k, z, zd = entity._best_component(
                            sample[:-1], p, groups=groups
                        )
                        mean = np.asarray(p["measurement"]["pose"]["means"])[k]
                        std = np.sqrt(
                            np.asarray(p["measurement"]["pose"]["covariances"])[k]
                        )
                        mean_raw = entity.model_to_value(mean)
                        residual = read[:3] - mean_raw[:3]
                        per_dim = [
                            float(np.sqrt(np.sum(zd[g] ** 2))) for g in groups
                        ]
                        logger.info(
                            f"  {label:8s} read {np.round(read[:3], 3)} "
                            f"component {k} mean {np.round(mean_raw[:3], 3)} "
                            f"delta {np.round(residual, 3)} m, sigma "
                            f"{np.round(std[:3], 3)}, z={z:.2f} "
                            f"{'OK' if z <= 4.03 else 'REJECT'} "
                            f"(groups {np.round(per_dim, 2)})"
                        )

                compare("pre / start read", pair.pre, start)
                compare("post / goal read", pair.post, goal)

                # the same pixels in both frames: the camera is fixed and the
                # buttons do not move, so a button read at a different pixel means
                # the matcher, and the same pixel with a different metre delta means
                # the depth or the extrinsics behind it
                if hasattr(reader, "kp_extractor"):
                    for name, frame in (("start", x_frame), ("goal", y_frame)):
                        if frame.rgb.numel() == 0:
                            logger.warning(f"  {name}: no frame to read")
                            continue
                        reader.reset_tracking()
                        poses = reader.kp_extractor.extract_poses(frame)
                        listed = " ".join(
                            f"{label}=({kp.x},{kp.y}) s={kp.score:.2f}"
                            for label, kp in poses.items()
                        )
                        logger.info(f"  {name} pixels: {listed}")

        graph.build(start, goal, 0.0)
        options = len(graph.ns_option.items)
        gated = int(graph.ns_option.gated.sum()) if options else options
        if graph.dead_end:
            rejected += 1
            hint = graph.dead_end_hint or "no entity reported a reason"
            logger.warning(f"episode {episode}: no option applies -- {hint}")
            for part in hint.split(" | "):
                key = part.split(" ")[0]
                reasons[key] = reasons.get(key, 0) + 1
        else:
            logger.info(
                f"episode {episode}: {options - gated}/{options} option(s) available"
            )

    logger.info(
        f"=== {args.episodes - rejected}/{args.episodes} episodes usable, "
        f"{rejected} discarded"
    )
    if reasons:
        ranked = sorted(reasons.items(), key=lambda item: -item[1])
        logger.warning(f"most frequent rejections: {ranked}")
    raise SystemExit(1 if rejected else 0)


if __name__ == "__main__":
    main()
