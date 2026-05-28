"""
Run a small number of explain steps with a trained GNN agent and visualize
the Integrated-Gradients node-importance attributions for each decision.

Usage (same config as train.py):
    python scripts/explain.py --config-name <cfg> [overrides...]
"""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from dataclasses import dataclass
from omegaconf import OmegaConf, SCMode

from tapas_gmm_modified.utils.argparse import parse_and_build_config
from src.agents.ppo.gnn import GNNAgent
from src.modules.buffer import BufferConfig, Buffer
from src.modules.storage import Storage, StorageConfig
from src.environments.environment import EnvironmentConfig
from src.agents.agent import AgentConfig
from src.experiments.experiment import ExperimentConfig
from src.modules.evaluators.evaluator import EvaluatorConfig
from src.factory import (
    select_agent,
    select_environment,
    select_experiment,
    select_evaluator,
)

N_STEPS = 10
OUTPUT_DIR = "results/explanations"


@dataclass
class ExplainConfig:
    agent: AgentConfig
    buffer: BufferConfig
    storage: StorageConfig
    evaluator: EvaluatorConfig
    experiment: ExperimentConfig
    environment: EnvironmentConfig


def _node_importance(mask) -> np.ndarray:
    """Sum feature-dim of a node mask to get per-node scalar importance."""
    if mask is None:
        return np.array([])
    t = mask.detach().cpu()
    if t.dim() == 2:
        return t.abs().sum(dim=-1).numpy()
    return t.abs().numpy()


def visualize_explanation(
    step_idx: int,
    skill_name: str,
    actor_expl,
    critic_expl,
    state_names: list[str],
    skill_names: list[str],
    save_dir: str,
):
    """
    Save a side-by-side bar-chart of node importances for the actor and critic
    explainer outputs.

    Node types and their labels:
      - goal   → state_names
      - obs    → state_names
      - task   → skill_names
      - actor  → skill_names  (actor head only)
      - critic → ["value"]    (critic head only)
    """
    node_label_map = {
        "goal": state_names,
        "obs": state_names,
        "task": skill_names,
        "actor": skill_names,
        "critic": ["value"],
    }

    def extract(expl) -> dict[str, np.ndarray]:
        if hasattr(expl, "node_mask_dict"):  # HeteroExplanation
            return {k: _node_importance(v) for k, v in expl.node_mask_dict.items()}
        if hasattr(expl, "node_mask") and expl.node_mask is not None:
            # Flat explanation – create a single entry
            return {"nodes": _node_importance(expl.node_mask)}
        return {}

    actor_masks = extract(actor_expl)
    critic_masks = extract(critic_expl)

    all_types = sorted(set(actor_masks) | set(critic_masks))
    if not all_types:
        print(f"[step {step_idx}] No node masks available, skipping plot.")
        return

    fig, axes = plt.subplots(
        len(all_types),
        2,
        figsize=(14, max(3 * len(all_types), 4)),
        squeeze=False,
    )
    fig.suptitle(f"Step {step_idx}  –  chosen skill: {skill_name}", fontsize=13)

    for row, ntype in enumerate(all_types):
        labels = node_label_map.get(ntype, None)

        for col, (masks, title) in enumerate(
            [(actor_masks, "Actor"), (critic_masks, "Critic")]
        ):
            ax = axes[row][col]
            imp = masks.get(ntype, np.array([]))

            if imp.size == 0:
                ax.set_visible(False)
                continue

            xs = np.arange(len(imp))
            ax.bar(xs, imp, color="steelblue" if col == 0 else "darkorange")
            ax.set_ylabel("Importance")
            ax.set_title(f"{title} – {ntype}")

            if labels and len(labels) == len(imp):
                ax.set_xticks(xs)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
            else:
                ax.set_xticks(xs)
                ax.set_xticklabels([str(i) for i in xs], fontsize=8)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"step_{step_idx:03d}_{skill_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved → {path}")


def run_explain(config: ExplainConfig, n_steps: int = N_STEPS):
    storage = Storage(config.storage)
    buffer = Buffer(config.buffer)

    evaluator = select_evaluator(config.evaluator, storage)
    env = select_environment(config.environment, evaluator, storage)
    experiment = select_experiment(config.experiment, env, storage)
    agent = select_agent(config.agent, storage, buffer)

    if not isinstance(agent, GNNAgent):
        raise TypeError(
            f"explain.py only supports GNNAgent, got {type(agent).__name__}"
        )

    # Put the network in eval mode so argmax is used instead of sampling
    agent.policy_old.eval()

    state_names = [s.name for s in agent.policy_old.states]
    skill_names = [s.name for s in agent.policy_old.skills]

    save_dir = os.path.join(OUTPUT_DIR, storage.config.tag)

    print(f"Running {n_steps} explain steps, saving plots to '{save_dir}' …")

    obs, goal = experiment.sample_task()
    episode_ended = False
    step = 0

    while step < n_steps:
        if episode_ended:
            obs, goal = experiment.sample_task()
            episode_ended = False

        skill, actor_expl, critic_expl = agent.explain(obs, goal)

        print(f"[step {step}] skill = {skill.name}")
        visualize_explanation(
            step_idx=step,
            skill_name=skill.name,
            actor_expl=actor_expl,
            critic_expl=critic_expl,
            state_names=state_names,
            skill_names=skill_names,
            save_dir=save_dir,
        )

        obs, reward, done, episode_ended = experiment.step(skill)
        agent.feedback(reward, done, episode_ended)
        step += 1

    experiment.close()
    print("Done.")


def entry_point():
    _, dict_config = parse_and_build_config(data_load=False, need_task=False)

    dict_config["storage"]["tag"] = (
        dict_config["storage"]["tag"]
        + f"_pe{dict_config['experiment']['p_empty']}_pr{dict_config['experiment']['p_rand']}"
    )

    config = OmegaConf.to_container(
        dict_config, resolve=True, structured_config_mode=SCMode.INSTANTIATE
    )
    run_explain(config)  # type: ignore


if __name__ == "__main__":
    entry_point()
