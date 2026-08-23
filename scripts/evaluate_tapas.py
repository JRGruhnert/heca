import argparse
import json
import sys
from collections import Counter
from pathlib import Path

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

from heca.data.data import DCEntity, DCScene
from heca.experts.expert import ExpertModel
from heca.misc import logger

try:
    from scripts.common.args import add_scene_argument, add_tag_argument
    from scripts.common.scenes import agents_by_scene
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Could not import the scene/agent configs: {exc}\n"
        "Make sure the scene dependencies (e.g. the ogbench fork pinned in "
        "pyproject.toml) are installed in the active environment."
    ) from exc


def sample_condition_scene(
    con: Condition, entities: dict[str, Entity], extras: dict
) -> DCScene:
    """Sample one value per entity from a condition's fitted models."""
    dc: dict[str, DCEntity] = {}
    for label, entity in entities.items():
        value = con.models[label].sample(1)[0]
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
        value = np.asarray(value).squeeze()
        value = entity.model_to_value(value)
        dc[label] = DCEntity(value=value, feature=entity.gnn_format(value))
    return DCScene(dc, extras=extras)


def condition_info_dict(
    con: Condition, entities: dict[str, Entity], extras: dict
) -> dict:
    scene = sample_condition_scene(con, entities, extras)
    info: dict = {}
    for label in con.models.keys():
        info.update(entities[label].env_state_value(label, scene))
    return info


def evaluate_agent(
    agent: ExpertModel, scene: OGScene, episodes: int, max_tries: int
) -> Counter:
    env = scene.env.unwrapped
    con = agent.conditions
    labels = set(con.pre.models) | set(con.post.models)
    entities = {label: scene.entities[label] for label in labels}

    counts: Counter = Counter()
    for ep in range(episodes):
        scene.env.reset(options={"render_goal": True})
        extras = scene.get_extras(env.compute_ob_info())
        pre_info = condition_info_dict(con.pre, entities, extras)
        post_info = condition_info_dict(con.post, entities, extras)

        # Teleport exactly once per episode; retries continue from the state
        # the failed attempt ended in.
        env.set_start(pre_info)
        env.set_goal(post_info)
        y = scene.to_dc_scene(env.get_reset_info()["goal"])

        succeeded_on = 0
        for attempt in range(1, max_tries + 1):
            # Current scene: teleported pre state on the 1st try, the end
            # state of the previous (failed) rollout afterwards.
            x = scene.to_dc_scene(env.compute_ob_info())
            _, fb = agent.act(x, y)
            if fb.reward == 1.0:
                succeeded_on = attempt
                break

        counts[succeeded_on] += 1
        logger.debug(
            f"[{scene.cfg.tag}] {agent.cfg.tag} episode {ep + 1}/{episodes}: "
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
    add_tag_argument(parser)
    parser.add_argument(
        "--episodes",
        type=int,
        default=100,
        help="Evaluation episodes per agent (each runs up to --max-tries "
        "rollouts); default 100 so the per-attempt statistics are meaningful.",
    )
    parser.add_argument(
        "--max-tries",
        type=int,
        default=5,
        help="Max attempts per episode before giving up (default: 5).",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Directory for the plots/JSON (default: <scene save dir>/plots).",
    )
    parser.add_argument("--seed", type=int, default=0, help="RNG seed.")
    args = parser.parse_args()
    np.random.seed(args.seed)

    n_failures = 0
    for scene_tag, models in agents_by_scene().items():
        if args.scene and scene_tag != args.scene:
            continue
        logger.info(f"[{scene_tag}] evaluating {len(models)} agents")

        results: list[dict] = []
        failures: list[dict] = []
        scene = None
        for cfg in models:
            if args.tag and cfg.tag != args.tag:
                continue
            try:
                agent = ExpertModel.get(cfg, auto_load=False)
                agent.load()
                scene = agent.scene
                counts = evaluate_agent(agent, scene, args.episodes, args.max_tries)
                results.append(
                    {"scene": scene_tag, "tag": cfg.tag, "counts": dict(counts)}
                )
                total = sum(counts.values())
                ok = total - counts.get(0, 0)
                hist = ", ".join(
                    f"try{t}={counts.get(t, 0)}" for t in range(1, args.max_tries + 1)
                )
                logger.info(
                    f"[{scene_tag}] {cfg.tag}: {ok}/{total} episodes ok "
                    f"({hist}, failed={counts.get(0, 0)})"
                )
            except Exception as exc:  # keep going if one agent fails
                logger.error(f"[{scene_tag}] agent {cfg.tag} failed: {exc}")
                failures.append({"scene": scene_tag, "tag": cfg.tag, "error": str(exc)})
                n_failures += 1

        if not results:
            logger.warning(
                f"[{scene_tag}] no agents evaluated "
                "(missing demos/policies? run pipeline_03 first)."
            )
            continue

        out_dir = Path(args.out) if args.out else scene.save_dir(scene.cfg) / "plots"
        out_dir.mkdir(parents=True, exist_ok=True)
        plot_path = plot_scene(
            scene_tag, results, out_dir, args.max_tries, args.episodes
        )
        json_path = out_dir / f"eval_success_{scene_tag}.json"
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
        logger.info(f"[{scene_tag}] wrote {plot_path}")
        scene.close()

    if n_failures:
        logger.warning(f"{n_failures} agent(s) failed; see per-scene JSON reports.")
        sys.exit(1)


if __name__ == "__main__":
    main()
