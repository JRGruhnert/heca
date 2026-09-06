import argparse

import matplotlib

matplotlib.use("Agg")  # headless plotting

from heca.experts.tapas import TapasExpert
from heca.misc import logger

from scripts.common.args import add_model_argument, add_scene_argument, add_tag_argument
from scripts.common.scenes import iter_agents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    parser.add_argument(
        "--max-demos",
        "--max_demos",
        dest="max_demos",
        type=int,
        default=30,
        help="Only plot the first N demos of each agent.",
    )
    args = parser.parse_args()

    for cfg in iter_agents():
        if args.scene and cfg.scene.tag != args.scene:
            continue
        if args.model and cfg.tag != args.model:
            continue
        logger.info(f"[{cfg.scene.tag}] Plotting demo velocities for {cfg.tag}")
        agent = TapasExpert.get(cfg, auto_load=False)
        agent.plot_demo_velocities(max_demos=args.max_demos)


if __name__ == "__main__":
    main()
