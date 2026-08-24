import argparse

from heca.guis.demo_selector import TapasDemoSelector

from scripts.common.args import add_scene_argument, add_tag_argument
from scripts.common.scenes import find_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="scene0")
    add_tag_argument(parser, default="close_drawer")
    args = parser.parse_args()
    agent_cfg = find_model(scene_tag=args.scene, model_tag=args.model)
    selector = TapasDemoSelector.get(TapasDemoSelector.Config(agent=agent_cfg))
    selector.run()


if __name__ == "__main__":
    main()
