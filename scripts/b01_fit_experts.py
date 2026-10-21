import argparse
import multiprocessing as mp
import signal
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

# make ``conf`` and ``scripts.common`` importable when run as a file, not as -m
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt

from heca.experts.tapas import TapasExpert
from heca.learning import dist as pdist
from heca.misc import logger

from scripts.common.args import (
    add_encoder_arguments,
    add_model_argument,
    add_scene_argument,
    add_threads_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import (
    experts_for_scene,
    scene_config,
    scene_tags,
    select_encoder,
)


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


def prepare_scene_extras(
    models: list[ExpertModel.Config], gt: bool, derive_extras: bool = True
) -> None:
    for cfg in models:
        model = ExpertModel.get(cfg, auto_load=False)
        model.use_gt(gt)
        model.load_extras(derive_extras)
    if models:
        ExpertModel.get(models[0], auto_load=False).scene.finish_extras()


def pipeline_scene(
    scene_cfg: Scene.Config,
    models: list[ExpertModel.Config],
    gt: bool,
    episodes: int,
    max_tries: int,
    skip_fit: bool = False,
    skip_conditions: bool = False,
    skip_eval: bool = False,
    condition_demos: int = 0,
):
    """Run the full fit/evaluate pipeline for every model of one scene."""
    prepare_scene_extras(models, gt, derive_extras=not skip_conditions)
    for cfg in models:
        logger.info(f"[{scene_cfg.tag}] === pipeline for {cfg.tag} ===")
        model = ExpertModel.get(cfg, auto_load=False)
        model.use_gt(gt)
        assert isinstance(model, TapasExpert), "Only Tapas is supported atm."
        if not skip_fit:
            fit_tapas(model)
        if not skip_conditions:
            model.fit_conditions(condition_demos)
        if not skip_eval:
            # imported here so a fit-only run does not need the plotting module
            from scripts.b03_plot_tapas_models import evaluate_one

            evaluate_one(cfg, scene_cfg, episodes, max_tries, gt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_encoder_arguments(parser)
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
        "--skip-fit",
        action="store_true",
        help="Keep the policies on disk and only build the conditions.",
    )
    parser.add_argument(
        "--skip-conditions",
        action="store_true",
        help="Only fit the policies; the conditions need the state readout and "
        "can be built on their own later.",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Fit (and build conditions) without stage 3.",
    )
    parser.add_argument(
        "--condition-demos",
        type=int,
        default=0,
        help="Demos to build the conditions from (0 = every demo in the file).",
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
        # before the configs go to the workers, so they carry the choice
        select_encoder(tag, args.backbone, args.stride)
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
    # spawn, not fork: hardware.device initialises CUDA when this module is
    # imported (it asks the cards how much memory they have), and a forked child
    # cannot re-initialise it -- "Cannot re-initialize CUDA in forked subprocess"
    pool = ProcessPoolExecutor(
        max_workers=workers,
        initializer=partial(pdist.pin_intra_op_threads, threads),
        mp_context=mp.get_context("spawn"),
    )
    futures = [
        pool.submit(
            pipeline_scene,
            scene_cfg,
            models,
            args.gt,
            args.episodes,
            args.max_tries,
            args.skip_fit,
            args.skip_conditions,
            args.skip_eval,
            args.condition_demos,
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
