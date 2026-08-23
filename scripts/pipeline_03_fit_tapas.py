"""Fit a TapasExpert for every configured agent.

Replicates the per-agent notebook workflow (load demos, fit stage 1, fit
stage 2, save) for all agents across the scene conf modules, in a single run.
The demos loaded for each agent come from its own ``segment_ids`` config value.
"""

import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt

from heca.experts.tapas import TapasExpert
from heca.misc import logger

from scripts.common.args import add_ee_argument, add_scene_argument, add_tag_argument
from scripts.common.scenes import iter_agents, to_ee


def _safe_avg(values):
    out = []
    for v in values:
        v = float(v)
        out.append(v if math.isfinite(v) else None)
    return out


def fit_agent(cfg: TapasExpert.Config):
    logger.info(
        f"[{cfg.scene.tag}] Fitting agent {cfg.tag} (segment_ids={cfg.segment_ids})"
    )
    agent = TapasExpert.get(cfg, auto_load=False)
    demos = agent.load_demos()

    _, avg_loglik_1 = agent.fit_stage1(demos)
    save_plots(agent, "fit_stage1")  # velocity-segmentation debug figures
    agent.plot_stage1()
    save_plots(agent, "stage1")

    _, avg_loglik_2 = agent.fit_stage2(demos)
    save_plots(agent, "fit_stage2")
    agent.plot_stage2()
    save_plots(agent, "stage2")

    agent.save()

    save_log(
        agent,
        {
            "scene": cfg.scene.tag,
            "tag": cfg.tag,
            "segment_ids": list(cfg.segment_ids) if cfg.segment_ids else [],
            "stage1_avg_loglik": _safe_avg(avg_loglik_1),
            "stage2_avg_loglik": _safe_avg(avg_loglik_2),
        },
    )
    logger.info(f"[{cfg.scene.tag}] Done agent {cfg.tag}")


def save_plots(agent: TapasExpert, stage: str):
    """Save all figures produced by the last plot call to disk."""
    out_dir = TapasExpert.save_dir(agent.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    for num in plt.get_fignums():
        fig = plt.figure(num)
        path = out_dir / f"{stage}_{num}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved plot to {path}")
    plt.close("all")


def save_log(agent: TapasExpert, record: dict):
    """Write the fit result to this agent's log subfolder."""
    out_dir = TapasExpert.save_dir(agent.cfg) / "log"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "fit_log.json"
    path.write_text(json.dumps(record, indent=2))
    logger.info(f"Wrote fit log to {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_tag_argument(parser)
    add_ee_argument(parser)
    args = parser.parse_args()

    for cfg in iter_agents():
        if args.scene and cfg.scene.tag != args.scene:
            continue
        if args.tag and cfg.tag != args.tag:
            continue
        if args.ee:
            cfg = to_ee(cfg)
        fit_agent(cfg)


if __name__ == "__main__":
    main()
