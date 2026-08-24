import argparse

from heca.scenes.scene import Scene

from scripts.common.args import add_scene_argument
from scripts.common.scenes import iter_scene_configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    args = parser.parse_args()

    for scene_cfg in iter_scene_configs():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        scene = Scene.get(scene_cfg)
        scene.demo_auto_extract()


if __name__ == "__main__":
    main()
