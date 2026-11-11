import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import cast

from heca.data.condition import Condition
from heca.data.entity import Entity
from heca.data.pair import ConPair
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
                label,
                dcscene,
                unnormalize_pos=scene.unnormalize_position,
            )
        )
    return info


def anchor_entities(conditions: ConPair) -> set[str]:
    return {
        label
        for label, score in conditions.change_scores.items()
        if score < Entity.ANCHOR_THRESHOLD
    }


def sample_task_conditions(
    pair: ConPair, scene: Scene, anchors: set[str]
) -> tuple[dict, dict]:
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


def save_prediction_plots(model: ExpertModel, episode: int, attempt: int) -> int:
    """Persist the figures a prediction batch opened (TAPAS ``dbg_prediction``)."""
    out_dir = ExpertModel.save_dir(model.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for num in plt.get_fignums():
        figure = plt.figure(num)
        path = (
            out_dir
            / f"{model.cfg.tag}_prediction_ep{episode:02d}_try{attempt:02d}_{num}.png"
        )
        figure.savefig(path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction plot to {path}")
        written += 1
    plt.close("all")
    return written


def save_model_plots(model: ExpertModel, demo_idx: int = 0, extended: bool = False) -> int:
    """The model's own diagnostics, each call guarded and saved on its own."""
    out_dir = ExpertModel.save_dir(model.cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    inner = model.model
    demos = getattr(inner, "_demos", None)

    def capture(label: str) -> int:
        written = 0
        for num in plt.get_fignums():
            figure = plt.figure(num)
            path = out_dir / f"{model.cfg.tag}_model_{label}_{num}.png"
            figure.savefig(path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved model plot to {path}")
            written += 1
        plt.close("all")
        return written

    calls = {
        "model": lambda: inner.plot_model(
            scatter=True,
            annotate_gaussians=True,
            annotate_trajs=True,
            mean_as_base=False,
            time_based=True,
        ),
        "hmm": inner.plot_hmm_transition_matrix,
    }
    if extended:
        # these need state a loaded model no longer carries (they are built
        # during fit_trajectories), so they are opt-in and may only warn
        calls |= {
            "components": lambda: _model_components(inner, demos),
            "basis": inner.plot_basis,
            "states": lambda: inner.plot_state_sequence_for_demo(demo_idx=demo_idx),
        }
    written = 0
    for label, call in calls.items():
        try:
            call()
        except Exception as exc:  # a plot that needs absent state must not stop the run
            logger.warning(f"{model.cfg.tag}: model plot {label!r} failed ({exc})")
            plt.close("all")
            continue
        written += capture(label)
    logger.info(f"{model.cfg.tag}: {written} model plot figure(s) written")
    return written


def _model_components(inner, demos) -> None:
    """``plot_model_components`` needs the joint models, recomputed from the demos."""
    if demos is None:
        raise RuntimeError("the model carries no demos to rebuild the joint models")
    joint_models = None
    trans_marginals = None
    _, trans_marginals, joint_models = inner.get_marginals_and_joint(
        fix_frames=True,
        time_based=True,
        heal_time_variance=True,
        frame_trans=demos._frame_origins_fixed.numpy(),
        frame_quats=demos._frame_origins_fixed_wquats.numpy(),
    )
    inner.plot_model_components(joint_models, trans_marginals)
    inner.plot_model_frames(joint_models, trans_marginals)


def predict_only(
    cfg: ExpertModel.Config,
    scene_cfg: Scene.Config,
    batches: int,
    gt: bool,
    model_plots: bool = False,
    demo_idx: int = 0,
    model_plots_all: bool = False,
) -> int:
    """Capture prediction batches without running a success test.

    The success evaluation is the expensive part (episodes x attempts of env
    stepping); a prediction batch is a single forward prediction from one
    observation. This resets the env, sets a start and a goal, asks the policy
    for one batch per reset and saves the figures -- no scoring, no results
    json, no chart.
    """
    cfg.policy.dbg_prediction = True  # the batch plot is opened by the policy
    model = ExpertModel.get(cfg, auto_load=False)
    model.use_gt(gt)
    model.load()
    model.load_extras()
    model.use_states = False
    if model_plots or model_plots_all:
        save_model_plots(model, demo_idx, extended=model_plots_all)

    scene = model.scene
    if gt:
        conditions = model.conditions
    else:
        # the task itself always comes from the ground-truth conditions (see
        # evaluate_one); only the observation the policy sees is rendered
        model.__dict__.pop("conditions", None)
        model._use_gt = True
        try:
            conditions = model.conditions
        finally:
            model._use_gt = gt

    env = get_env_safely(scene)
    anchors = anchor_entities(conditions)
    visual = not model._use_gt
    saved = 0
    for index in range(batches):
        env.reset(options={"render_goal": True})
        pre_info, post_info = sample_task_conditions(conditions, scene, anchors)
        env.set_start(pre_info)
        env.set_goal(post_info)
        goal_info = dict(env.get_reset_info()["goal"])
        y = scene.to_dc_scene(goal_info)
        if visual:
            env._render_goal = True
            env.set_goal(goal_info, return_info=False)
            rendered = getattr(env, "_cur_goal_rendered", None)
            if rendered is not None:
                goal_info["image"] = rendered
                y = model.make_scene(y, scene.to_td_image(goal_info))
            env.set_start(pre_info)
        ob_info = env.compute_ob_info()
        x = scene.to_dc_scene(ob_info)
        if visual:
            x = model.make_scene(x, scene.render_frame())
        logger.info(f"[{scene_cfg.tag}] {cfg.tag}: predicting batch {index + 1}/{batches}")
        model.act(x, y)
        written = save_prediction_plots(model, index, 1)
        saved += written
        if written == 0:
            logger.warning(
                f"{cfg.tag}: the policy opened no figure for batch {index + 1}; "
                "the model may not take the batch-prediction path"
            )
    logger.info(f"[{scene_cfg.tag}] {cfg.tag}: {saved} prediction figure(s) written")
    return saved


def evaluate_model(
    model: ExpertModel,
    scene: Scene,
    episodes: int,
    max_tries: int,
    conditions: ConPair,
    capture: int = 0,
) -> Counter:
    """``capture`` is the number of prediction batches to save as figures."""
    budget = int(capture)
    env = get_env_safely(scene)
    counts: Counter = Counter()
    anchors = anchor_entities(conditions)
    # a visual policy reads images, so both scenes it plans from are the rendered
    # frames through its own encoder -- the GT poses would be a different frame
    # than the one it was fitted in. The task instance still comes from the
    # conditions above.
    visual = not model._use_gt
    began = time.perf_counter()
    for ep in range(episodes):
        env.reset(options={"render_goal": True})
        pre_info, post_info = sample_task_conditions(conditions, scene, anchors)

        env.set_start(pre_info)
        env.set_goal(post_info)
        goal_info = dict(env.get_reset_info()["goal"])
        y = scene.to_dc_scene(goal_info)
        if visual:
            # render the goal from the state just set, the way _sample_task does
            env._render_goal = True
            env.set_goal(goal_info, return_info=False)
            rendered = getattr(env, "_cur_goal_rendered", None)
            if rendered is None:
                logger.warning(
                    f"[{scene.cfg.tag}] no goal render available; the goal stays "
                    "ground truth for this episode"
                )
            else:
                goal_info["image"] = rendered
                y = model.make_scene(y, scene.to_td_image(goal_info))
            env.set_start(pre_info)  # the goal render may have moved the state

        succeeded_on = 0
        for attempt in range(1, max_tries + 1):
            ob_info = env.compute_ob_info()
            x = scene.to_dc_scene(ob_info)
            if visual:
                # compute_ob_info carries no image, so the frame is rendered
                x = model.make_scene(x, scene.render_frame())
            _, fb = model.act(x, y)
            if budget > 0:
                budget -= save_prediction_plots(model, ep, attempt) or 1
            if fb.success:
                succeeded_on = attempt
                break

        counts[succeeded_on] += 1
        elapsed = time.perf_counter() - began
        left = elapsed / (ep + 1) * (episodes - ep - 1)
        logger.info(
            f"[{scene.cfg.tag}] {model.cfg.tag} episode {ep + 1}/{episodes}: "
            f"{'success on try ' + str(succeeded_on) if succeeded_on else 'failed'}"
            f" (counts {dict(counts)}, ~{left / 60:.0f} min left)"
        )
    return counts


def _plot_scene_impl(
    scene_tag: str,
    results: list[dict],
    out_dir: Path,
    max_tries: int,
    episodes: int,
    stem: str = "",
) -> Path:
    results = [
        {**r, "counts": {int(k): v for k, v in r["counts"].items()}} for r in results
    ]
    tags = [r["tag"] for r in results]
    n = len(tags)
    pct = np.zeros((n, max_tries + 1))
    for i, r in enumerate(results):
        c = r["counts"]
        total = sum(c.values()) or 1
        for t in range(1, max_tries + 1):
            pct[i, t - 1] = 100.0 * c.get(t, 0) / total
        pct[i, max_tries] = 100.0 * c.get(0, 0) / total  # failed / gave up

    fig, ax = plt.subplots(figsize=(max(4.5, 0.6 * n), 4.2))
    xpos = np.arange(n)
    bottom = np.zeros(n)
    cmap = plt.get_cmap("summer")
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
        color="1.0",
    )

    ax.set_xticks(xpos)
    ax.set_xticklabels(tags, rotation=45, ha="right", fontsize=8)
    ax.set_ylim(0, 102)
    ax.set_ylabel("episodes [%]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

    fig.tight_layout()
    path = out_dir / f"{stem or f'eval_success_{scene_tag}'}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_scene(
    scene_tag: str,
    results: list[dict],
    out_dir: Path,
    max_tries: int,
    episodes: int,
    stem: str = "",
) -> Path:
    """Stacked bar chart of per-attempt success rates."""
    return _plot_scene_impl(scene_tag, results, out_dir, max_tries, episodes, stem)


def results_path(scene_cfg: Scene.Config, suffix: str = "") -> Path:
    """Shared, stateless results file for one scene and keypoint backend."""
    out_dir = Scene.save_dir(scene_cfg) / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)
    tail = f"-{suffix}" if suffix else ""
    return out_dir / f"eval_success_{scene_cfg.tag}{tail}.json"


def load_results(json_path: Path) -> dict:
    if json_path.exists():
        try:
            return json.loads(json_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def update_results(json_path: Path, entry: dict, episodes: int, max_tries: int):
    """Insert/replace one agent's entry in the scene results json (in place).

    Stateless: reads the current file, updates only this model's record, writes
    back. Safe to call per model so partial/interrupted runs keep their state.
    """
    data = load_results(json_path)
    data["episodes"] = episodes
    data["max_tries"] = max_tries
    agents = data.setdefault("agents", [])
    for i, agent in enumerate(agents):
        if agent.get("tag") == entry["tag"]:
            agents[i] = entry
            break
    else:
        agents.append(entry)
    data.setdefault("failures", [])
    json_path.write_text(json.dumps(data, indent=2))
    return data


def plot_results(
    scene_cfg: Scene.Config, json_path: Path | None = None, suffix: str = ""
) -> Path | None:
    """(Re)plot the current scene chart from the persisted results json."""
    json_path = json_path or results_path(scene_cfg, suffix)
    data = load_results(json_path)
    agents = data.get("agents", [])
    if not agents:
        return None
    episodes = int(data.get("episodes", 100))
    max_tries = int(data.get("max_tries", 3))
    return plot_scene(
        scene_cfg.tag, agents, json_path.parent, max_tries, episodes, json_path.stem
    )


def backend_suffix(cfg: ExpertModel.Config) -> str:
    return str(getattr(cfg.kp_extraction, "backend", "dino"))


def evaluate_one(
    cfg: ExpertModel.Config,
    scene_cfg: Scene.Config,
    episodes: int,
    max_tries: int,
    gt: bool,
    capture: int = 0,
    model_plots: bool = False,
    demo_idx: int = 0,
    model_plots_all: bool = False,
) -> Counter:
    """Evaluate one model, persist its result, re-plot the scene chart."""
    # the policy reads this when it predicts a whole batch; set before load()
    cfg.policy.dbg_prediction = capture > 0
    # select the variant first: `load()` picks tapas_gt.pt or tapas_img.pt from it
    model = ExpertModel.get(cfg, auto_load=False)
    model.use_gt(gt)
    model.load()
    model.load_extras()
    # a visual rollout re-encodes the frame on every env step: the keypoints are
    # needed (the policy reads positions), the state readout is not -- it never
    # reaches the TPGMM, and the policies were fitted with it off as well. Asking
    # Molmo per step is what made episodes take minutes.
    model.use_states = False
    if model_plots or model_plots_all:
        save_model_plots(model, demo_idx, extended=model_plots_all)
    # the task instance is sampled from the *ground-truth* conditions even for a
    # visual policy: a visual condition describes the tracked keypoint, and its
    # position block is that keypoint while its rotation block is hard zero, so
    # turning one into a reset state puts the object in the wrong place, facing the
    # wrong way. The policy still acts on images; only the task definition is GT.
    if gt:
        conditions = model.conditions
    else:
        # fetched without touching the visual pair: that one needs Molmo
        model.__dict__.pop("conditions", None)
        model._use_gt = True
        try:
            conditions = model.conditions
        finally:
            model._use_gt = gt
    counts = evaluate_model(
        model, model.scene, episodes, max_tries, conditions, capture
    )

    json_path = results_path(scene_cfg, backend_suffix(cfg))
    update_results(
        json_path,
        {"scene": scene_cfg.tag, "tag": cfg.tag, "counts": dict(counts)},
        episodes,
        max_tries,
    )
    plot_path = plot_results(scene_cfg, json_path)

    total = sum(counts.values())
    ok = total - counts.get(0, 0)
    hist = ", ".join(f"try{t}={counts.get(t, 0)}" for t in range(1, max_tries + 1))
    logger.info(
        f"[{scene_cfg.tag}] {cfg.tag}: {ok}/{total} episodes ok "
        f"({hist}, failed={counts.get(0, 0)})"
    )
    if plot_path is not None:
        logger.info(f"[{scene_cfg.tag}] updated {plot_path}")
    return counts


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
    parser.add_argument(
        "--dbg-prediction",
        type=int,
        default=0,
        metavar="N",
        help="Save N TAPAS prediction-batch plots (dbg_prediction), one per "
        "attempt, into the model's plots/ folder (0 = off). Raise --episodes to "
        "collect them from more than one episode.",
    )
    parser.add_argument(
        "--predict-only",
        type=int,
        default=0,
        metavar="N",
        help="Only capture N prediction-batch plots: no success evaluation, no "
        "results json, no chart.",
    )
    parser.add_argument(
        "--model-plots",
        action="store_true",
        help="Also save the model's own diagnostics (plot_model, HMM "
        "transition matrices) into the model's plots/ folder.",
    )
    parser.add_argument(
        "--model-plots-all",
        action="store_true",
        help="Also try the plots that need fit-time state (components, basis, "
        "state sequence); on a loaded model these usually only warn.",
    )
    parser.add_argument(
        "--demo-idx",
        type=int,
        default=0,
        help="Demo index for the state-sequence plot (default 0).",
    )
    parser.add_argument(
        "--replot",
        action="store_true",
        help="Skip evaluation: re-plot the chart from the persisted results jsons.",
    )

    args = parser.parse_args()

    if args.replot:
        for scene_cfg, _ in agents_by_scene():
            if args.scene and scene_cfg.tag != args.scene:
                continue
            out_dir = Scene.save_dir(scene_cfg) / "plots"
            jsons = sorted(out_dir.glob(f"eval_success_{scene_cfg.tag}*.json"))
            for json_path in jsons:
                path = plot_results(scene_cfg, json_path)
                if path is not None:
                    logger.info(f"[{scene_cfg.tag}] replotted {path} from {json_path.name}")
                else:
                    logger.info(f"[{scene_cfg.tag}] {json_path.name}: no agents to plot")
        return

    for scene_cfg, models in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        if args.viewer:
            Scene.get(scene_cfg, auto_load=False).cfg.viewer = True
        logger.info(f"[{scene_cfg.tag}] evaluating {len(models)} agents")

        for cfg in models:
            if args.model and cfg.tag != args.model:
                continue
            if args.predict_only:
                predict_only(
                    cfg,
                    scene_cfg,
                    args.predict_only,
                    args.gt,
                    args.model_plots,
                    args.demo_idx,
                    args.model_plots_all,
                )
                continue
            evaluate_one(
                cfg,
                scene_cfg,
                args.episodes,
                args.max_tries,
                args.gt,
                args.dbg_prediction,
                args.model_plots,
                args.demo_idx,
                args.model_plots_all,
            )


if __name__ == "__main__":
    main()
