from dataclasses import dataclass, field
from abc import abstractmethod
from pathlib import Path
from torch import nn
import torch
from torch.distributions import Categorical
from hoopgn.environments.environment import Environment
from hoopgn.misc import logger
from hoopgn.hoops.hoop import Hoop
from hoopgn.agents.agent import Agent
from hoopgn.misc.td import TDScene


class MPNetwork(Hoop):
    @dataclass(kw_only=True)
    class Config(Hoop.Config):
        environment: Environment.Query

        label: str = field(init=False)
        checkpoint_path: str | None = None
        eval_mode: bool = False
        dim_encoder: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        if self.cfg.checkpoint_path:
            self.load()

        if self.cfg.eval_mode:
            self.eval()

    @property
    def dim_property(self) -> int:
        raise NotImplementedError()

    @property
    def dim_agent(self) -> int:
        raise NotImplementedError()

    def encode(self, x: TDScene, y: TDScene) -> tuple[TDScene, TDScene]:
        return x, y

    def predict(self, x: TDScene, y: TDScene) -> Agent:
        with torch.no_grad():
            batch = self.to_encoded_batch(x, y)
            logits, value = self.forward(batch)
        assert logits.shape == (
            1,
            self.dim_agent,
        ), f"Expected logits shape (1, {self.dim_agent}), got {logits.shape}"
        assert value.shape == (1,), f"Expected value shape ({1},), got {value.shape}"

        dist = Categorical(logits=logits)
        if self.training:
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        logprob = dist.log_prob(action)
        self.buffer.act_values(x, y, action.detach(), logprob.detach(), value.detach())
        return self.skills[int(action.item())]

    @abstractmethod
    def forward(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement the forward method.")

    @abstractmethod
    def explain(self, batch, index: int) -> tuple:
        raise NotImplementedError("This network does not support explanations.")

    def _encode_properties(self, x: torch.Tensor, label: str) -> torch.Tensor:
        return self.encoder.get(label)(x)

    def _pre_encode_properties(self, x: TDScene) -> torch.Tensor:
        temp = []
        for property in self.properties:
            pre = property.postprocess(x[property.cfg.label])
            enc = self._encode_properties(pre, property.cfg.encoder.label)
            temp.append(enc)
        return torch.stack(temp, dim=0)

    def to_batch(self, x: list[TDScene], y: list[TDScene]) -> torch.Tensor:
        assert len(x) == len(y), "Current and goal lists must have the same length."
        current_encoded = [self._pre_encode_properties(x) for x in x]
        goal_encoded = [self._pre_encode_properties(x) for x in y]
        return self._to_batch(current_encoded, goal_encoded, x)

    @abstractmethod
    def _to_batch(
        self, x: list[torch.Tensor], y: list[torch.Tensor], obs: list[TDScene]
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abstractmethod
    def load(self):
        raise NotImplementedError()

    def save(self, highscore: bool, index: int):
        if highscore:
            tag = Path("highscore_epoch{}.pt".format(index))
        else:
            tag = Path("checkpoint_epoch{}.pt".format(index))

        logger.info(f"Saving weights to: {self.path / tag} for epoch {index}")
        torch.save({"state": self.state_dict()}, self.path / tag)
