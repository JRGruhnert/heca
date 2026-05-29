from dataclasses import dataclass
from loguru import logger
import torch
from src.observation.observation import StateValueDict
from src.agents.ppo.ppo import PPOAgent, PPOAgentConfig
from src.hardware import device

from src.modules.buffer import Buffer
from src.modules.storage import Storage
from src.networks.gnn.gnn4 import Explanation, Gnn
from src.skills.skill import Skill


@dataclass
class GNNAgentConfig(PPOAgentConfig):
    # Add any GNN-specific configuration parameters here
    pass


class GNNAgent(PPOAgent):
    def __init__(
        self,
        config: GNNAgentConfig,
        storage: Storage,
        buffer: Buffer,
    ):
        super().__init__(
            config,
            Gnn(storage.states, storage.skills),
            Gnn(storage.states, storage.skills),
            buffer,
            storage,
        )
        self.load()

    def load(self):
        """
        Load the model from the specified path.
        """
        if self.storage.config.checkpoint_path is None:
            # No checkpoint specified
            return

        logger.info(
            "Loading GNN checkpoint from: {}".format(
                self.storage.config.checkpoint_path
            )
        )
        checkpoint = torch.load(
            self.storage.config.checkpoint_path, map_location=device
        )

        self.policy_old.load_state_dict(checkpoint["model_state"], strict=False)
        self.policy_new.load_state_dict(checkpoint["model_state"], strict=False)

    def explain(
        self,
        obs: StateValueDict,
        goal: StateValueDict,
        step: int,
        state_diff: dict[str, bool],
    ) -> Skill:
        import os
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        action, action_logprob, state_val, actor_expl, critic_expl = (
            self.policy_old.explain(obs, goal)
        )
        skill = self.storage.skills[int(action.item())]
        self.buffer.act_values(obs, goal, action, action_logprob, state_val)

        states = self.policy_old.states
        skills = self.policy_old.skills
        state_names = [s.name for s in states]
        skill_names = [s.name for s in skills]

        # --- Per-node IG importance (sum over encoder-feature dim) ---
        def node_imp(expl, node_type):
            nd = expl.node_mask_dict if hasattr(expl, "node_mask_dict") else {}
            m = nd.get(node_type)
            if m is None:
                return np.zeros(0)
            return m.detach().cpu().abs().sum(dim=-1).numpy()

        # --- Combined figure (3 rows × 2 cols) ---
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle(f"Step {step} – Chosen skill: {skill.name}", fontsize=14)
        gs = fig.add_gridspec(3, 2, hspace=0.6, wspace=0.45)

        # Row 0: which states differ between obs and goal
        ax_cmp = fig.add_subplot(gs[0, :])
        diff_vals = np.array([int(state_diff.get(n, False)) for n in state_names])
        xs = np.arange(len(state_names))
        ax_cmp.bar(xs, diff_vals, color="steelblue")
        ax_cmp.set_xticks(xs)
        ax_cmp.set_xticklabels(state_names, rotation=45, ha="right", fontsize=7)
        ax_cmp.set_yticks([0, 1])
        ax_cmp.set_yticklabels(["same", "different"])
        ax_cmp.set_title("States that differ between obs and goal")

        # Rows 1–2: goal and obs node importances for actor (col 0) and critic (col 1)
        for col, (expl, head) in enumerate(
            [(actor_expl, "Actor"), (critic_expl, "Critic")]
        ):
            color = "steelblue" if col == 0 else "darkorange"
            for row_off, ntype in enumerate(["goal", "obs"]):
                imp = node_imp(expl, ntype)
                ax = fig.add_subplot(gs[1 + row_off, col])
                if imp.size == 0:
                    ax.set_visible(False)
                    continue
                labels = (
                    state_names
                    if len(imp) == len(state_names)
                    else [str(i) for i in range(len(imp))]
                )
                xi = np.arange(len(imp))
                ax.bar(xi, imp, color=color)
                ax.set_xticks(xi)
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
                ax.set_title(f"{head} – {ntype} node importance")
                ax.set_ylabel("IG attribution")

        save_dir = "results/explanations"
        os.makedirs(save_dir, exist_ok=True)
        combined = os.path.join(save_dir, f"{step:02d}_{skill.name}.png")
        fig.savefig(combined, dpi=150, bbox_inches="tight")
        plt.close(fig)

        node_label_map = {
            "goal": state_names,
            "obs": state_names,
            "task": skill_names,
            "actor": skill_names,
            "critic": ["value"],
        }
        for expl, tag in [(actor_expl, "actor"), (critic_expl, "critic")]:
            # aggregate [num_nodes, 32] → [num_nodes] scalar for visualize_graph
            for ntype in expl.node_types:
                m = expl[ntype].get("node_mask")
                if m is not None and m.dim() == 2:
                    expl[ntype].node_mask = m.abs().sum(dim=-1, keepdim=True)
            nd = expl.node_mask_dict if hasattr(expl, "node_mask_dict") else {}
            labels = {k: node_label_map[k] for k in nd if k in node_label_map}
            expl.visualize_graph(
                path=os.path.join(
                    save_dir, f"step_{step:03d}_{skill.name}_{tag}_graph.png"
                ),
                node_labels=labels,
            )
        return skill
