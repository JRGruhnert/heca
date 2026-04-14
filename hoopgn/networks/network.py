from dataclasses import dataclass, field
from typing import Any
import torch
from abc import ABC, abstractmethod
from hoopgn.networks.layers.encoder import (
    PropertyEncoderRegistry,
    PropertyEncoderRegistryConfig,
)

from hoopgn.observation.td_properties import TDProperties
from hoopgn.skills.skill import Skill
from hoopgn.properties.property import Property
from hoopgn import hardware, logger


@dataclass(kw_only=True)
class NetworkConfig:
    dim_skill: int
    dim_state: int
    label: str = field(init=False)
    checkpoint_path: str | None = None
    registry: PropertyEncoderRegistryConfig = PropertyEncoderRegistryConfig()


class Network(torch.nn.Module, ABC):

    def __init__(
        self,
        config: NetworkConfig,
    ):
        super().__init__()
        self.config = config
        self.encoder = PropertyEncoderRegistry(config.registry)
        self.skills: list[Skill] = []
        self.properties: list[Property] = []

    def register_encoder(self, properties: list[Property]):
        self.properties = properties
        for state in properties:
            self.encoder.register(state.config.encoder)

    def register_skills(self, skills: list[Skill]):
        self.skills = skills

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def explain(self, batch, skill: Skill) -> tuple:
        raise NotImplementedError("This network does not support explanations.")

    def _encode_properties(self, x: torch.Tensor, label: str) -> torch.Tensor:
        return self.encoder.get(label)(x)

    def _pre_encode_properties(
        self, x: TDProperties, states: list[Property]
    ) -> torch.Tensor:
        temp = []
        for state in states:
            pre = state.modify(x[state.config.label])
            enc = self._encode_properties(pre, state.config.encoder.label)
            temp.append(enc)
        return torch.stack(temp, dim=0)

    def to_encoded_batch(
        self,
        current: list[TDProperties] | TDProperties,
        goal: list[TDProperties] | TDProperties,
        states: list[Property],
    ) -> torch.Tensor:
        """Converts lists of observations and goals into a batch suitable for the network."""
        if isinstance(current, TDProperties):
            current = [current]
        if isinstance(goal, TDProperties):
            goal = [goal]
        assert len(current) == len(
            goal
        ), "Current and goal lists must have the same length."

        current_encoded = [self._pre_encode_properties(x, states) for x in current]
        goal_encoded = [self._pre_encode_properties(x, states) for x in goal]

        return self._to_batch(current_encoded, goal_encoded, current)

    @abstractmethod
    def _to_batch(
        self,
        current: list[torch.Tensor],
        goal: list[torch.Tensor],
        obs: list[TDProperties],
    ) -> torch.Tensor:
        raise NotImplementedError("Subclasses must implement the _to_batch method.")

    def load(self, skills: list[Skill], states: list[Property]):
        self.register_encoder(states)
        self.register_skills(skills)
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
    def _load(self, checkpoint: Any, skills: list[Skill], states: list[Property]):
        raise NotImplementedError("Subclasses must implement the _load method.")

    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        return torch.load(checkpoint_path, map_location=hardware.device)
