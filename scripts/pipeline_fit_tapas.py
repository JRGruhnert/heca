"""Fit a TapasExpert for every configured agent.

Replicates the per-agent notebook workflow (load demos, fit stage 1, fit
stage 2, save) for all agents across the scene conf modules, in a single run.
The demos loaded for each agent come from its own ``segment_ids`` config value.
"""

import argparse

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt

from heca.experts.tapas import TapasExpert
from heca.misc import logger

import conf.experts.scene1
import conf.experts.scene2
import conf.experts.scene3
import conf.experts.scene4
import conf.experts.scene5
import conf.experts.sceneog


def all_agents():
    for mod in (
        conf.experts.scene1,
        conf.experts.scene2,
        conf.experts.scene3,
        conf.experts.scene4,
        conf.experts.scene5,
        conf.experts.sceneog,
    ):
        for cfg in mod.agents:
            yield mod.__name__.split(".")[-1], cfg


def fit_agent(scene: str, cfg: TapasExpert.Config):
    tag = cfg.tag
    logger.info(f"[{scene}] Fitting agent {tag} (segment_ids={cfg.segment_ids})")
    agent = TapasExpert.get(cfg)
    demos = agent.load_demos()
    agent.fit_stage1(demos)
    agent.plot_stage1()
    save_plots(agent, "stage1")
    agent.fit_stage2(demos)
    agent.plot_stage2()
    save_plots(agent, "stage2")
    agent.save()
    logger.info(f"[{scene}] Done agent {tag}")


def save_plots(agent: TapasExpert, stage: str):
    """Save all figures produced by the last plot call to disk."""
    out_dir = TapasExpert.save_dir(agent.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = out_dir / f"{stage}_{num}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene",
        help="Only fit agents from this scene module (e.g. sceneog). "
        "Defaults to all scenes.",
    )
    parser.add_argument(
        "--tag",
        help="Only fit the agent with this tag.",
    )
    args = parser.parse_args()

    for scene, cfg in all_agents():
        if args.scene and scene != args.scene:
            continue
        if args.tag and cfg.tag != args.tag:
            continue
        fit_agent(scene, cfg)


if __name__ == "__main__":
    main()
