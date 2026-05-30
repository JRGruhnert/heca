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

        save_dir = "results/explanations"
        for expl, tag in [(actor_expl, "actor"), (critic_expl, "critic")]:
            expl.visualize_feature_importance(
                path=os.path.join(
                    save_dir,
                    f"step_{step:03d}_{skill.name}_{tag}_feature_importance.png",
                )
            )
            expl.visualize_graph(
                path=os.path.join(
                    save_dir, f"step_{step:03d}_{skill.name}_{tag}_graph.png"
                ),
                backend="networkx",
            )
        return skill
