"""Interactively select demos for a TapasExpert.

Launches the demo-selector GUI for the agent identified by ``--scene`` and
``--tag``.
"""

import argparse

from heca.guis.demo_selector import TapasDemoSelector

from scripts.common.args import add_scene_argument, add_tag_argument
from scripts.common.scenes import find_agent


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="sceneog")
    add_tag_argument(parser, default="close_drawer")
    args = parser.parse_args()

    agent_cfg = find_agent(scene_tag=args.scene, tag=args.tag)
    if agent_cfg is None:
        parser.error(f"No agent found for scene={args.scene!r} tag={args.tag!r}")

    selector = TapasDemoSelector.get(TapasDemoSelector.Config(agent=agent_cfg))
    selector.run()


if __name__ == "__main__":
    main()
