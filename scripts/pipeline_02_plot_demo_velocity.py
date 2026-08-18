import argparse

import matplotlib

matplotlib.use("Agg")  # headless plotting

from heca.experts.tapas import TapasExpert
from heca.misc import logger

import conf.experts.scene1
import conf.experts.scene2
import conf.experts.scene3
import conf.experts.scene4
import conf.experts.scene5
import conf.experts.sceneog


def all_agents():
    for mod in (
        # conf.experts.scene1,
        # conf.experts.scene2,
        # conf.experts.scene3,
        # conf.experts.scene4,
        # conf.experts.scene5,
        conf.experts.sceneog,
    ):
        for cfg in mod.agents:
            yield cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        help="Only plot agents from this scene module (e.g. sceneog). "
        "Defaults to all scenes.",
    )
    parser.add_argument(
        "--tag",
        help="Only plot the agent with this tag.",
    )
    parser.add_argument(
        "--max-demos",
        type=int,
        default=None,
        help="Only plot the first N demos of each agent.",
    )
    args = parser.parse_args()

    for cfg in all_agents():
        if args.scene and cfg.scene.tag != args.scene:
            continue
        if args.tag and cfg.tag != args.tag:
            continue
        logger.info(f"[{cfg.scene.tag}] Plotting demo velocities for {cfg.tag}")
        agent = TapasExpert.get(cfg, auto_load=False)
        agent.plot_demo_velocities(max_demos=args.max_demos)


if __name__ == "__main__":
    main()
