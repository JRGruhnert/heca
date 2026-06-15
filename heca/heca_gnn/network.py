from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

from heca.misc.td import TDWorld
from heca.misc import hardware, logger
from heca.misc.base import Persistable
from heca.heca_gnn.actor import ActorReadoutNetwork
from heca.heca_gnn.bases.base import BaseNetwork
from heca.heca_gnn.critic import CriticReadoutNetwork
from torch_geometric.data import HeteroData


class HecaNetwork(Persistable, nn.Module):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        config: "HecaNetwork.Config"
        base: BaseNetwork.Config
        feature_dim: int = 64
        eval_mode: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.base = BaseNetwork.get(self.cfg.base)
        self.actor_net = ActorReadoutNetwork(feature_dim=self.cfg.feature_dim)
        self.critic_net = CriticReadoutNetwork(feature_dim=self.cfg.feature_dim)

        if self.cfg.eval_mode:
            self.eval()

    def encode(self, x: TDWorld, y: TDWorld) -> tuple[TDWorld, TDWorld]:
        raise NotImplementedError()

    def forward(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        return self.actor(
            HeteroData(x=x, x_dict=x_dict, edge_index_dict=edge_index_dict)
        )

    def actor(self, data: HeteroData) -> torch.Tensor:
        return self.actor_net(data)

    def critic(self, data: HeteroData) -> torch.Tensor:
        return self.critic_net(data)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    # @classmethod
    # def load(cls, query: "HecaNetwork.Query") -> "HecaNetwork":
    #     file = cls.File.get_latest(query)
    #     model = cls.get(query)
    #     if file:
    #         logger.info(f"Loading weights from: {file}")
    #         ckpt = torch.load(file, map_location=hardware.device)
    #         model.load_state_dict(ckpt["model_state"], strict=False)
    #     return model

    # @classmethod
    # def save(cls, query: "HecaNetwork.Query", tag: str, epoch: int) -> bool:
    #     path = cls.File.resolve(query)
    #     file = cls.File.file_name(epoch, tag)
    #     torch.save({"model_state": cls.get(query).state_dict()}, path / file)
    #     logger.info(f"Saved model for epoch {epoch} with tag: {tag}.")
    #     return True
