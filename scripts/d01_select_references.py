import argparse

from heca.guis.scene_sample_selector import SceneRefSelector

from scripts.common.args import add_scene_argument
from scripts.common.scenes import find_scene_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    parser.add_argument(
        "--dataset",
        default="",
        help="demo .h5 to take the frames from; default: the scene's own",
    )
    parser.add_argument(
        "--samples", type=int, default=5, help="state pictures per state"
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    if scene_cfg is None:
        parser.error(f"No scene found with tag {args.scene!r}")

    selector = SceneRefSelector.get(
        SceneRefSelector.Config(
            scene=scene_cfg,
            dataset_name=args.dataset,
            sample_count=args.samples,
        )
    )
    selector.run()


if __name__ == "__main__":
    main()
