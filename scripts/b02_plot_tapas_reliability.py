import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import cast

from heca.conditions.condition import Condition
from heca.data.entity import Entity
from heca.scenes.ogbench.scene import OGScene
from heca.scenes.scene import Scene

# Make ``conf`` / ``scripts.common`` importable when run directly as
# ``python scripts/evaluate_tapas.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np
import torch
from ogbench.manipspace.envs.scene_env_base import SceneEnvBase
from heca.data.data import DCEntity, DCScene
from heca.experts.expert import ExpertModel
from heca.misc import logger
from scripts.common.args import (
    add_model_argument,
    add_scene_argument,
    add_tag_argument,
    add_use_gt_argument,
    add_viewer_argument,
)
from scripts.common.scenes import agents_by_scene


def sample_dcscene(con: Condition) -> DCScene:
    """Sample one value per entity from a condition's fitted models."""
    dc: dict[str, DCEntity] = {}
    for label, entity in con.entities.items():
        value = con.models[label].sample(1)[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value).squeeze()
        value = entity.model_to_value(value)
        dc[label] = DCEntity(value=value, feature=entity.gnn_format(value))
    return DCScene(dc)


def condition_info_dict(con: Condition, scene: Scene) -> dict:
    dcscene = sample_dcscene(con)
    info: dict = {}
    for label, entity in con.entities.items():
        info.update(
            entity.env_state_value(
                label, dcscene, unnormalize_pos=scene.unnormalize_position
            )
        )
    return info


def anchor_entities(agent: ExpertModel) -> set[str]:
    """Entities that do not move in this agent's task (change score below the
    anchor threshold)."""
    return {
        label
        for label, score in agent.conditions.change_scores.items()
        if score < Entity.ANCHOR_THRESHOLD
    }


def sample_task_conditions(
    agent: ExpertModel, scene: Scene, anchors: set[str]
) -> tuple[dict, dict]:
    """Sample a consistent (pre, post) task pair.

    Anchor entities (e.g. the button in a faucet task — it only determines
    whether the handle is unlocked) never move in the demos. Sampling pre and
    post independently would demand a goal state the policy cannot produce, so
    for anchors the post value is set equal to the sampled pre value; only
    entities the agent actually changes get their own sampled post value.
    """
    pair = agent.conditions
    pre_info = condition_info_dict(pair.pre, scene)
    post_info = condition_info_dict(pair.post, scene)
    for label in anchors:
        for key in list(post_info):
            if key.startswith(f"heca_{label}_"):
                post_info[key] = pre_info[key]
    return pre_info, post_info


def get_env_safely(scene: Scene) -> SceneEnvBase:
    assert isinstance(scene, OGScene), "Only OgScene Supported."
    return cast(SceneEnvBase, scene.env.unwrapped)


def evaluate_model(
    model: ExpertModel,
    scene: Scene,
    episodes: int,
    max_tries: int,
) -> Counter:
    env = get_env_safely(scene)
    counts: Counter = Counter()
    anchors = anchor_entities(model)
    for ep in range(episodes):
        env.reset(options={"render_goal": True})
        pre_info, post_info = sample_task_conditions(model, scene, anchors)

        env.set_start(pre_info)
        env.set_goal(post_info)
        y = scene.to_dc_scene(env.get_reset_info()["goal"])

        succeeded_on = 0
        for attempt in range(1, max_tries + 1):
            x = scene.to_dc_scene(env.compute_ob_info())
            _, fb = model.act(x, y)
            if fb.reward == 1.0:
                succeeded_on = attempt
                break

        counts[succeeded_on] += 1
        logger.debug(
            f"[{scene.cfg.tag}] {model.cfg.tag} episode {ep + 1}/{episodes}: "
            f"{'success on try ' + str(succeeded_on) if succeeded_on else 'failed'}"
        )
    return counts


def plot_scene(
    scene_tag: str,
    results: list[dict],
    out_dir: Path,
    max_tries: int,
    episodes: int,
) -> Path:
    """Stacked bar chart of per-attempt success rates for one scene."""
    tags = [r["tag"] for r in results]
    n = len(tags)
    pct = np.zeros((n, max_tries + 1))
    for i, r in enumerate(results):
        c = r["counts"]
        total = sum(c.values()) or 1
        for t in range(1, max_tries + 1):
            pct[i, t - 1] = 100.0 * c.get(t, 0) / total
        pct[i, max_tries] = 100.0 * c.get(0, 0) / total  # failed / gave up

    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * n), 6.0))
    xpos = np.arange(n)
    bottom = np.zeros(n)
    cmap = plt.get_cmap("viridis")
    colors = [cmap(k / max(1, max_tries - 1)) for k in range(max_tries)]
    for t in range(max_tries):
        ax.bar(
            xpos,
            pct[:, t],
            bottom=bottom,
            width=0.65,
            label=f"{t + 1}. try",
            color=colors[t],
        )
        bottom += pct[:, t]
    ax.bar(
        xpos,
        pct[:, max_tries],
        bottom=bottom,
        width=0.65,
        label="failed",
        color="0.35",
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("episodes [%]")
    ax.set_title(f"Tapas success reliability — {scene_tag} ({episodes} episodes/agent)")
    ax.legend(loc="upper right", fontsize=8)

    # Total success rate above each bar (success on any attempt).
    for i, r in enumerate(results):
        c = r["counts"]
        total = sum(c.values())
        ok = total - c.get(0, 0)
        ax.text(
            i,
            101.5,
            f"{100.0 * ok / total:.0f}%",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    fig.tight_layout()
    path = out_dir / f"eval_success_{scene_tag}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_viewer_argument(parser)
    add_use_gt_argument(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=3,
        help="Max attempts per episode before giving up.",
    )

    args = parser.parse_args()

    for scene_cfg, models in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        if args.viewer:
            Scene.get(scene_cfg, auto_load=False).cfg.viewer = True
        logger.info(f"[{scene_cfg.tag}] evaluating {len(models)} agents")

        results: list[dict] = []
        failures: list[dict] = []
        for cfg in models:
            if args.model and cfg.tag != args.tag:
                continue

            model = ExpertModel.get(cfg).use_gt(args.gt)
            counts = evaluate_model(
                model,
                model.scene,
                args.episodes,
                args.max_tries,
            )
            results.append(
                {"scene": scene_cfg.tag, "tag": cfg.tag, "counts": dict(counts)}
            )
            total = sum(counts.values())
            ok = total - counts.get(0, 0)
            hist = ", ".join(
                f"try{t}={counts.get(t, 0)}" for t in range(1, args.max_tries + 1)
            )
            logger.info(
                f"[{scene_cfg.tag}] {cfg.tag}: {ok}/{total} episodes ok "
                f"({hist}, failed={counts.get(0, 0)})"
            )

        out_dir = Scene.save_dir(scene_cfg) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_scene(
            scene_cfg.tag, results, out_dir, args.max_tries, args.episodes
        )
        json_path = out_dir / f"eval_success_{scene_cfg.tag}.json"
        json_path.write_text(
            json.dumps(
                {
                    "episodes": args.episodes,
                    "max_tries": args.max_tries,
                    "agents": results,
                    "failures": failures,
                },
                indent=2,
            )
        )
        logger.info(f"[{scene_cfg.tag}] wrote {plot_path}")


if __name__ == "__main__":
    main()
