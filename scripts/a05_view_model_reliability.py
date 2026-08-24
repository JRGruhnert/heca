import argparse

from heca.guis.tapas_agent_tester import TapasManualExecuter

from scripts.common.args import add_scene_argument
from scripts.common.scenes import find_scene_config, find_scene_models


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="scene1")
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth observations (default: true).",
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)

    agents = find_scene_models(args.scene)

    tester_cfg = TapasManualExecuter.Config(
        agents=agents,
        scene=scene_cfg,
        use_gt=args.use_gt,
    )

    tester = TapasManualExecuter.get(tester_cfg)
    tester.run()


if __name__ == "__main__":
    main()
