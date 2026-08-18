"""Segment and extract demos for every configured scene.

Replicates the per-scene demo extraction (``scene.demo_auto_extract``) for all
scenes referenced by the scene conf modules, in a single run.
"""

import argparse

from heca.scenes.scene import Scene

import conf.experts.scene1
import conf.experts.scene2
import conf.experts.scene3
import conf.experts.scene4
import conf.experts.scene5
import conf.experts.sceneog

SCENE_MODULES = (
    conf.experts.scene1,
    conf.experts.scene2,
    conf.experts.scene3,
    conf.experts.scene4,
    conf.experts.scene5,
    # conf.experts.sceneog,
)


def all_scenes():
    seen = set()
    for mod in SCENE_MODULES:
        for agent_cfg in mod.agents:
            scene_cfg = agent_cfg.scene
            if scene_cfg.tag in seen:
                continue
            seen.add(scene_cfg.tag)
            yield scene_cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        help="Only segment demos for this scene module (e.g. sceneog). "
        "Defaults to all scenes.",
    )
    args = parser.parse_args()

    for scene_cfg in all_scenes():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        scene = Scene.get(scene_cfg)
        scene.demo_auto_extract()


if __name__ == "__main__":
    main()
