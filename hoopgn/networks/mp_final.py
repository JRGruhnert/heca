from dataclasses import dataclass, field
from torch import nn
from abc import abstractmethod

import torch
from hoopgn.agents.leaf_agent import LeafAgent
from hoopgn.base import ConfigurableClass
from hoopgn.environments.environment import Environment
from torch.distributions import Categorical
from hoopgn.observation.td_properties import TDProperties
from hoopgn.agents.agent import Agent


class MPNetwork(ConfigurableClass, nn.Module):
    @dataclass(kw_only=True)
    class Config(ConfigurableClass.Config):
        environment: Environment.Signature

        label: str = field(init=False)
        checkpoint_path: str | None = None
        eval_mode: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        if self.cfg.checkpoint_path:
            self.load()

        if self.cfg.eval_mode:
            self.eval()

    @property
    def dim_state(self) -> int:
        return len(Environment.get(self.cfg.environment).properties)

    @property
    def dim_skill(self) -> int:
        skills = set()
        for _, leaf_agent in LeafAgent.registry.items():
            leaf_agent = LeafAgent.get(leaf_agent.signature)
            if leaf_agent.cfg.sig.environment == self.cfg.environment:
                skills.add(leaf_agent.cfg.sig)
        return len(skills)

    def predict(self, obs: TDProperties, goal: TDProperties) -> Agent:
        with torch.no_grad():
            batch = self.to_encoded_batch(obs, goal)
            logits, value = self.forward(batch)
        assert logits.shape == (
            1,
            self.dim_skill,
        ), f"Expected logits shape (1, {self.dim_skill}), got {logits.shape}"
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
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
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
        self, x: list[torch.Tensor], y: list[torch.Tensor], obs: list[TDProperties]
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()
