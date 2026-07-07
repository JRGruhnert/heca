from dataclasses import dataclass
import os
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
    ) -> Skill:
        assert isinstance(self.policy_old, Gnn), "Policy must be Gnn for explanation"
        action, action_logprob, state_val, actor_expl, critic_expl = (
            self.policy_old.explain(obs, goal)
        )
        skill = self.storage.skills[int(action.item())]
        self.buffer.act_values(obs, goal, action, action_logprob, state_val)

        state_names = [s.name for s in self.policy_old.states]
        skill_names = [s.name for s in self.policy_old.skills]
        save_dir = f"results/explanations/{self.storage.config.tag}"
        os.makedirs(save_dir, exist_ok=True)
        for expl, tag in [(actor_expl, "actor"), (critic_expl, "critic")]:
            for ntype in expl.node_types:
                m = expl[ntype].get("node_mask")
                if m is not None:
                    expl[ntype].node_mask = m.detach()
            expl.visualize_feature_importance(
                path=os.path.join(save_dir, f"{step:02d}_{skill.name}_{tag}_feat.png"),
                top_k=20,
            )
            # aggregate [N, 32] → [N, 1] for visualize_graph
            for ntype in expl.node_types:
                m = expl[ntype].get("node_mask")
                if m is not None and m.dim() == 2:
                    agg = m.abs().sum(dim=-1, keepdim=True)
                    expl[ntype].node_mask = agg / (agg.max() + 1e-8)
            expl.visualize_graph(
                path=os.path.join(save_dir, f"{step:02d}_{skill.name}_{tag}_graph.png"),
                node_labels={
                    "goal": state_names,
                    "obs": state_names,
                    "task": skill_names,
                    # "actor": skill_names,
                },
            )
        return skill
