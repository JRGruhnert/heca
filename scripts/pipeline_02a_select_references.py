"""Interactively select keypoint / state references for a scene.

Launches the reference-selector GUI for the scene identified by ``--scene``.
"""

import argparse

from heca.guis.scene_sample_selector import SceneRefSelector

from scripts.common.args import add_scene_argument
from scripts.common.scenes import find_scene_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="sceneog")
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    if scene_cfg is None:
        parser.error(f"No scene found with tag {args.scene!r}")

    selector = SceneRefSelector.get(SceneRefSelector.Config(scene=scene_cfg))
    selector.run()


if __name__ == "__main__":
    main()
