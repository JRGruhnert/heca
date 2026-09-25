import argparse
import signal
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import matplotlib

from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt

from heca.experts.tapas import TapasExpert
from heca.learning import dist as pdist
from heca.misc import logger

from scripts.b03_plot_tapas_models import evaluate_one
from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_threads_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import experts_for_scene, scene_config, scene_tags


def fit_tapas(expert: TapasExpert):
    demos = expert.load_demos()
    expert.fit_stage1(demos)
    save_plots(expert, "fit_stage1")  # velocity-segmentation debug figures
    expert.plot_stage1()
    save_plots(expert, "stage1")

    expert.fit_stage2(demos)
    save_plots(expert, "fit_stage2")
    expert.plot_stage2()
    save_plots(expert, "stage2")

    expert.save()


def save_plots(agent: TapasExpert, stage: str):
    out_dir = TapasExpert.save_dir(agent.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = out_dir / f"{stage}_{num}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")
    plt.close("all")


def pipeline_scene(
    scene_cfg: Scene.Config,
    models: list[ExpertModel.Config],
    gt: bool,
    episodes: int,
    max_tries: int,
):
    """Run the full fit/evaluate pipeline for every model of one scene."""
    for cfg in models:
        logger.info(f"[{scene_cfg.tag}] === pipeline for {cfg.tag} ===")
        model = ExpertModel.get(cfg, auto_load=False)
        model.use_gt(gt)
        assert isinstance(model, TapasExpert), "Only Tapas is supported atm."
        fit_tapas(model)
        model.fit_conditions()
        evaluate_one(cfg, scene_cfg, episodes, max_tries, gt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    add_threads_argument(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent (stage 3).",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Max attempts per episode before giving up (stage 3).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max scenes fitted concurrently (0 = one process per scene, 1 = sequential).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    jobs = []
    # filter on tags first, so a single-scene run imports only that scene
    for tag in scene_tags():
        if args.scene and tag != args.scene:
            continue
        models = [
            m
            for m in experts_for_scene(tag)
            if not (args.model and m.tag != args.model)
        ]
        if models:
            jobs.append((scene_config(tag), models))

    if not jobs:
        logger.error(
            f"No scene/model matches the given filters "
            f"(scene={args.scene!r}, model={args.model!r})."
        )
        raise SystemExit(1)

    workers = len(jobs) if args.jobs <= 0 else min(args.jobs, len(jobs))
    logger.info(f"Running pipeline for {len(jobs)} scenes with {workers} process(es).")
    threads = pdist.threads_for(workers, args.threads)
    logger.info(f"Threads per worker: {threads} (for {workers} process(es))")
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=partial(pdist.pin_intra_op_threads, threads),
    )
    futures = [
        pool.submit(
            pipeline_scene, scene_cfg, models, args.gt, args.episodes, args.max_tries
        )
        for scene_cfg, models in jobs
    ]

    try:
        for future in as_completed(futures):
            future.result()
    finally:
        # the workers share this process group, so an interrupt reaches them too
        pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
