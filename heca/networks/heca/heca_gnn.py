from dataclasses import dataclass
from pathlib import Path
from typing import Type
from tensordict import TensorDict
import torch
from torch import nn
from torch_geometric.data import HeteroData

from heca.properties.encoders.encoder import PropertyEncoder
from heca.misc import hardware, logger
from heca.misc.classes import StorageClass
from heca.misc.td import TDScene
from heca.networks.heca.actor import ActorReadoutNetwork
from heca.networks.heca.bases.base import BaseNetwork
from heca.networks.heca.critic import CriticReadoutNetwork
from torch_geometric.data import HeteroData
from typing import TypeVar, Type

V = TypeVar("V", bound="HecaGN")


class HecaGN(StorageClass, nn.Module):
    @dataclass(kw_only=True)
    class Query(StorageClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StorageClass.Config):
        query: "HecaGN.Query"
        base: BaseNetwork.Config
        encoders: set[PropertyEncoder.Query]
        feature_dim: int = 32
        eval_mode: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.encoders = nn.ModuleDict(
            {query.label: PropertyEncoder.search(query) for query in self.cfg.encoders}
        )
        self.base = BaseNetwork.from_config(self.cfg.base)
        self.actor_net = ActorReadoutNetwork(feature_dim=self.cfg.feature_dim)
        self.critic_net = CriticReadoutNetwork(feature_dim=self.cfg.feature_dim)

        if self.cfg.eval_mode:
            self.eval()

    def recursive_encode(self, tensor_dict: TDScene, prefix=""):
        for key, value in tensor_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, TensorDict):
                self.recursive_encode(value, prefix=full_key)
            else:
                query = PropertyEncoder.Query(label=full_key)  # Validate existence
                encoder = PropertyEncoder.search(query)
                tensor_dict[key] = encoder(value)

    def encode(self, x: TDScene, y: TDScene) -> tuple[TDScene, TDScene]:
        self.recursive_encode(x)
        self.recursive_encode(y)
        return x, y

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
    def resolve_path(
        cls: Type[V], query: "HecaGN.Query", epoch: int, label: str
    ) -> Path:
        return query.root / Path(query.label) / f"ckpt_{epoch}_{label}{query.ending}"

    def load(self, query: "HecaGN.Query", label: str = "") -> int:
        # Scan for all checkpoint files with the given label
        ckpt_dir = query.root / Path(query.label)
        pattern = f"ckpt_*_{label}{query.ending}"
        candidates = list(ckpt_dir.glob(pattern))
        if not candidates:
            raise FileNotFoundError(f"No checkpoint found for label '{label}'")

        # Extract epoch numbers and find the highest
        def extract_epoch(path):
            name = path.stem
            parts = name.split("_")
            try:
                return int(parts[1])
            except Exception:
                return -1

        candidates = [(extract_epoch(p), p) for p in candidates]
        candidates = [item for item in candidates if item[0] >= 0]
        if not candidates:
            raise FileNotFoundError(f"No valid checkpoint found for label '{label}'")
        epoch, path = max(candidates, key=lambda x: x[0])
        checkpoint = torch.load(path, map_location=hardware.device)
        self.load_state_dict(checkpoint["model_state"], strict=False)
        return epoch

    def save(self, query: "HecaGN.Query", epoch: int, label: str) -> None:
        path = self.resolve_path(query, epoch, label)
        logger.info(f"Saving weights to: {path}")
        torch.save({"model_state": self.state_dict()}, path)

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)
