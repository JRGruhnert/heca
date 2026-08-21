"""Manually execute TapasExperts in a scene.

Launches the manual-executer GUI for the agents of the scene identified by
``--scene``. Use ``--tag`` to run a single agent.
"""

import argparse

from heca.guis.tapas_agent_tester import TapasManualExecuter

from scripts.common.args import add_scene_argument, add_tag_argument
from scripts.common.scenes import agents_by_scene, find_scene_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="scene1")
    add_tag_argument(parser)
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth observations (default: true).",
    )
    parser.add_argument(
        "--passive-viewer",
        action="store_true",
        help="Use the MuJoCo passive viewer instead of matplotlib images.",
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    if scene_cfg is None:
        parser.error(f"No scene found with tag {args.scene!r}")

    agents = agents_by_scene().get(args.scene)
    if not agents:
        parser.error(f"No agents found for scene {args.scene!r}")

    if args.tag:
        agents = [cfg for cfg in agents if cfg.tag == args.tag]
        if not agents:
            parser.error(f"No agent found for scene={args.scene!r} tag={args.tag!r}")

    tester_cfg = TapasManualExecuter.Config(
        agents=agents,
        scene=scene_cfg,
        use_passive_viewer=args.passive_viewer,
        use_gt=args.use_gt,
    )

    tester = TapasManualExecuter.get(tester_cfg)
    tester.run()


if __name__ == "__main__":
    main()
