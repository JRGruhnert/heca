from dataclasses import dataclass
from typing import Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.networks.layers.encoder import StateEncoder
from src.observation.observation import StateValueDict
from src.states.state import State
from src.skills.skill import Skill
from loguru import logger

@dataclass
class NetworkConfig:
    checkpoint_path: str | None = None

class Network(nn.Module, ABC):

    def __init__(
        self,
        config: NetworkConfig,
        states: list[State],
        skills: list[Skill],
    ):
        super().__init__()
        self.config = config
        self.is_eval_mode = False
        self.states = states
        self.skills = skills
        self.dim_states = len(states)
        self.dim_skills = len(skills)
        self.dim_encoder = 32

        input_dims = {state.type: state.size for state in states}

        self.encoders = nn.ModuleDict(
            {
                type_str: StateEncoder(input_dim, self.dim_encoder)
                for type_str, input_dim in input_dims.items()
            }
        )

    def eval(self):
        super().eval()  # Call PyTorch's nn.Module.eval() instead of iterating manually
        self.is_eval_mode = True
        logger.info("Network set to evaluation mode.")

    @abstractmethod
    def forward(
        self,
        batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pass


    def _encode_states(self, x: StateValueDict) -> torch.Tensor:
        encoded_x = [
            self.encoders[state.type](state.make_input(x[state.name]))
            for state in self.states
        ]
        return torch.stack(encoded_x, dim=0)

    def to_encoded_batch(self, current: list[StateValueDict] | StateValueDict, goal: list[StateValueDict] | StateValueDict) -> torch.Tensor:
        """Converts lists of observations and goals into a batch suitable for the network."""
        if isinstance(current, StateValueDict):
            current = [current]
        if isinstance(goal, StateValueDict):
            goal = [goal]
        assert len(current) == len(goal), "Current and goal lists must have the same length."
        
        current_encoded = [self._encode_states(x) for x in current]
        goal_encoded = [self._encode_states(x) for x in goal]

        return self._to_batch(current_encoded, goal_encoded)

    @abstractmethod
    def _to_batch(self, current: list[torch.Tensor], goal: list[torch.Tensor]) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the _to_batch method.")
    
    def load(self):
        if self.config.checkpoint_path is not None:
            checkpoint = self._load_checkpoint(self.config.checkpoint_path)
            self._load(checkpoint)
            logger.info(f"Loading network checkpoint from: {self.config.checkpoint_path}")   
        else:
            logger.info("No checkpoint path provided in network config. Starting with a new model.")

    @abstractmethod
    def _load(self, checkpoint: Any):
         raise NotImplementedError("Subclasses must implement the _load method.")
    
    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        return torch.load(checkpoint_path, map_location=torch.device('cpu'))