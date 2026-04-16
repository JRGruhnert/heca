from dataclasses import dataclass, field
from typing import Any
import torch
from abc import ABC, abstractmethod
from hoopgn.networks.layers.encoder import (
    PropertyEncoderRegistry,
    PropertyEncoderRegistryConfig,
)
from torch.distributions import Categorical
from hoopgn.observation.td_properties import TDProperties
from hoopgn.agents.agent import Skill
from hoopgn.environments.properties.property import Property
from hoopgn import hardware, logger


@dataclass(kw_only=True)
class NetworkConfig:
    dim_skill: int
    dim_state: int
    label: str = field(init=False)
    checkpoint_path: str | None = None
    eval_mode: bool = False
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

        # Load checkpoint if path is provided in config
        if self.config.checkpoint_path:
            logger.info(f"Loading checkpoint from {self.config.checkpoint_path}.")
            checkpoint = self._load_checkpoint(self.config.checkpoint_path)
            self._load(checkpoint, self.skills, self.properties)

        if self.config.eval_mode:
            self.eval()

    def register_encoder(self, properties: list[Property]):
        self.properties = properties
        for state in properties:
            self.encoder.register(state.cfg.encoder)

    def register_skills(self, skills: list[Skill]):
        self.skills = skills

    def predict(
        self,
        obs: TDProperties,
        goal: TDProperties,
    ) -> Skill:
        with torch.no_grad():
            batch = self.to_encoded_batch(obs, goal)
            logits, value = self.forward(batch)
        assert logits.shape == (
            1,
            self.config.dim_skill,
        ), f"Expected logits shape (1, {self.config.dim_skill}), got {logits.shape}"
        assert value.shape == (1,), f"Expected value shape ({1},), got {value.shape}"

        dist = Categorical(logits=logits)
        if self.training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        logprob = dist.log_prob(action)
        self.buffer.act_values(
            obs, goal, action.detach(), logprob.detach(), value.detach()
        )
        return self.skills[int(action.item())]

    @abstractmethod
    def forward(
        self,
        *args,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def explain(self, batch, index: int) -> tuple:
        raise NotImplementedError("This network does not support explanations.")

    def _encode_properties(self, x: torch.Tensor, label: str) -> torch.Tensor:
        return self.encoder.get(label)(x)

    def _pre_encode_properties(self, x: TDProperties) -> torch.Tensor:
        temp = []
        for state in self.properties:
            pre = state.postprocess(x[state.cfg.label])
            enc = self._encode_properties(pre, state.cfg.encoder.label)
            temp.append(enc)
        return torch.stack(temp, dim=0)

    def to_batch(self, x: list[TDProperties], y: list[TDProperties]) -> torch.Tensor:
        assert len(x) == len(y), "Current and goal lists must have the same length."
        current_encoded = [self._pre_encode_properties(x) for x in x]
        goal_encoded = [self._pre_encode_properties(x) for x in y]
        return self._to_batch(current_encoded, goal_encoded, x)

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
        if self.config.checkpoint_path:
            logger.info(f"Loading network checkpoint.")
            checkpoint = self._load_checkpoint(self.config.checkpoint_path)
            self._load(checkpoint, skills, states)

        else:
            logger.warning("No checkpoint provided. Starting with untrained network.")

    @abstractmethod
    def _load(self, checkpoint: Any, skills: list[Skill], states: list[Property]):
        raise NotImplementedError("Subclasses must implement the _load method.")

    def _load_checkpoint(self, checkpoint_path: str) -> Any:
        return torch.load(checkpoint_path, map_location=hardware.device)
