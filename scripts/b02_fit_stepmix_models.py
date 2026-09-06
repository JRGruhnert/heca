import argparse

import matplotlib

matplotlib.use("Agg")  # headless plotting (ConPair.plot writes PNGs)

from heca.experts.tapas import TapasExpert
from heca.misc import logger

from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import iter_agents


def fit_conditions(cfg: TapasExpert.Config, gt: bool, rotation: bool) -> None:
    """Fit (and cache) the pre/post StepMix models for one rotation variant.

    A fresh, uncached instance is created per variant because ``conditions`` is
    a cached property and the rotation mode changes the fitted entity features.
    """
    variant = "with rotation" if rotation else "without rotation"
    logger.info(f"[{cfg.scene.tag}] Fitting conditions {variant}: {cfg.tag}")

    expert = TapasExpert(cfg)
    expert.use_gt(gt)
    expert.force_recompute()
    expert.load()

    # Accessing .conditions triggers the fit and writes the cache file.
    _ = expert.conditions

    cache = "conditions.joblib" if rotation else "conditions_pos.joblib"
    logger.info(f"[{cfg.scene.tag}] Done {cfg.tag} -> {expert.load_dir(cfg) / cache}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    args = parser.parse_args()

    for cfg in iter_agents():
        if args.scene and cfg.scene.tag != args.scene:
            continue
        if args.model and cfg.tag != args.model:
            continue
        assert isinstance(cfg, TapasExpert.Config), "Only Tapas is supported."
        fit_conditions(cfg, args.gt, rotation=True)
        fit_conditions(cfg, args.gt, rotation=False)


if __name__ == "__main__":
    main()
