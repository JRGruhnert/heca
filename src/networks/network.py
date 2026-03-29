from dataclasses import dataclass, field
from typing import Any
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from src.networks.layers.encoder import StateEncoder
from src.observation.observation import StateValueDict
from src.states.state import State
from src.skills.tree.leafs.leaf import Leaf
from loguru import logger


@dataclass
class StateTypeConfig:
    type: str
    size: int


@dataclass
class NetworkConfig:
    skill_count: int
    state_count: int
    name: str
    dim_encoder: int = 32
    checkpoint_path: str | None = None
    state_types: list[StateTypeConfig] = field(
        default_factory=lambda: [
            StateTypeConfig(type="EulerPrecise", size=3),
            StateTypeConfig(type="EulerArea", size=6),
            StateTypeConfig(type="Quat", size=4),
            StateTypeConfig(type="Range", size=1),
            StateTypeConfig(type="Bool", size=1),
            StateTypeConfig(type="Flip", size=1),
            StateTypeConfig(type="State", size=10),
        ]
    )


class Network(nn.Module, ABC):

    def __init__(
        self,
        config: NetworkConfig,
    ):
        super().__init__()
        self.config = config
        self.is_eval_mode = False

        self.encoders = nn.ModuleDict(
            {
                state.type: StateEncoder(state.size, self.config.dim_encoder)
                for state in config.state_types
            }
        )

    def eval(self):
        super().eval()  # Call PyTorch's nn.Module.eval() instead of iterating manually
        self.is_eval_mode = True
        logger.info("Network set to evaluation mode.")

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def explain(self, batch, skill: Leaf) -> Any:
        raise NotImplementedError("This network does not support explanations.")

    def _encode_states(self, x: StateValueDict, states: list[State]) -> torch.Tensor:
        encoded_x = [
            self.encoders[state.config.type_str](
                state.make_input(x[state.config.label])
            )
            for state in states
        ]
        return torch.stack(encoded_x, dim=0)

    def to_encoded_batch(
        self,
        current: list[StateValueDict] | StateValueDict,
        goal: list[StateValueDict] | StateValueDict,
        states: list[State],
    ) -> torch.Tensor:
        """Converts lists of observations and goals into a batch suitable for the network."""
        if isinstance(current, StateValueDict):
            current = [current]
        if isinstance(goal, StateValueDict):
            goal = [goal]
        assert len(current) == len(
            goal
        ), "Current and goal lists must have the same length."

        current_encoded = [self._encode_states(x, states) for x in current]
        goal_encoded = [self._encode_states(x, states) for x in goal]

        return self._to_batch(current_encoded, goal_encoded, current)

    @abstractmethod
    def _to_batch(
        self,
        current: list[torch.Tensor],
        goal: list[torch.Tensor],
        obs: list[StateValueDict],
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the _to_batch method.")

    def load(self, skills: list[Leaf], states: list[State]):
        if self.config.checkpoint_path is not None:
            checkpoint = self._load_checkpoint(self.config.checkpoint_path)
            self._load(checkpoint, skills, states)
            logger.info(
                f"Loading network checkpoint from: {self.config.checkpoint_path}"
            )
        else:
            logger.info(
                "No checkpoint path provided in network config. Starting with a new model."
            )

    @abstractmethod
    def _load(self, checkpoint: Any, skills: list[Leaf], states: list[State]):
        raise NotImplementedError("Subclasses must implement the _load method.")

    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        return torch.load(checkpoint_path, map_location=torch.device("cpu"))
