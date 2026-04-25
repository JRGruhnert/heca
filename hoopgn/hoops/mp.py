from dataclasses import dataclass, field
from abc import abstractmethod
from pathlib import Path
from torch import nn, Tensor
import torch
from hoopgn.environments.environment import Environment
from hoopgn.misc import logger
from hoopgn.hoops.hoop import Hoop
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
