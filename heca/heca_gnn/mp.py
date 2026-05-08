from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.heca_gnn.network import HecaNetwork
from heca.properties.encoders.encoder import PropertyEncoder
from heca.misc import hardware, logger
from heca.classes.persist import Persistable
from heca.misc.td import TDWorld
from heca.heca_gnn.actor import ActorReadoutNetwork
from heca.heca_gnn.bases.base import BaseNetwork
from heca.heca_gnn.critic import CriticReadoutNetwork
from torch_geometric.data import HeteroData
from typing import TypeVar

V = TypeVar("V", bound="HecaNetwork")


class MPNetwork(HecaNetwork):
    @dataclass(frozen=True, kw_only=True)
    class Query(HecaNetwork.Query):
        pass

    @dataclass(frozen=True, kw_only=True)
    class File(HecaNetwork.File):
        pass

        @classmethod
        def resolve_file_path(
            cls, query: "HecaNetwork.Query", epoch: int, tag: str
        ) -> Path:
            return cls.root / Path(query.label) / f"ckpt_{epoch}_{tag}_{cls.ending}"

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        query: "HecaNetwork.Query"
        base: BaseNetwork.Config
        feature_dim: int = 32
        eval_mode: bool = False
        encoders: list[PropertyEncoder.Query]

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.encoders = nn.ModuleDict(
            {
                encoder.label: PropertyEncoder.search(encoder)
                for encoder in self.cfg.encoders
            }
        )
        self.base = BaseNetwork.create(self.cfg.base)
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

    @classmethod
    def load(cls, query: "HecaNetwork.Query") -> "HecaNetwork":
        path = cls.Disk.get_latest(query)
        if path:
            logger.info(f"Loading weights from: {path}")
            ckpt = torch.load(path, map_location=hardware.device)
            model = cls.search(query)
            model.load_state_dict(ckpt["model_state"], strict=False)
            return model
        else:
            return cls.search(query)

    @classmethod
    def save(cls, query: "HecaNetwork.Query", epoch: int, tag: str) -> bool:
        path = cls.Disk.resolve_path(query, epoch, tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": cls.search(query).state_dict()}, path)
        logger.info(f"Saved {tag} for epoch {epoch}.")
        return True

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
