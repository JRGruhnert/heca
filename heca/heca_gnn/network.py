from dataclasses import dataclass
from pathlib import Path
import torch
from torch import nn

from heca.misc.td import TDWorld
from heca.misc import hardware, logger
from heca.classes.persist import Persistable
from heca.heca_gnn.actor import ActorReadoutNetwork
from heca.heca_gnn.bases.base import BaseNetwork
from heca.heca_gnn.critic import CriticReadoutNetwork
from torch_geometric.data import HeteroData


class HecaNetwork(Persistable, nn.Module):
    @dataclass(frozen=True, kw_only=True)
    class Query(Persistable.Query):
        pass

    @dataclass(frozen=True, kw_only=True)
    class File(Persistable.File):
        folder: str = "networks"
        ending: str = ".pt"

        @classmethod
        def file_name(cls, epoch: int, tag: str) -> str:
            assert tag in ["epoch", "best"]
            assert epoch >= 0
            return f"ckpt_{tag}_{epoch}{cls.ending}"

        @classmethod
        def find_matching_files(cls, directory: Path, pattern: str) -> list[Path]:
            return list(directory.glob(pattern))

        @classmethod
        def get_latest(cls, query: "HecaNetwork.Query") -> Path | None:
            directory = cls.resolve_directory(query)
            pattern = f"ckpt_epoch_*.pt"
            matching_files = cls.find_matching_files(directory, pattern)
            epochs = []
            for file in matching_files:
                try:
                    epoch_str = file.stem.split("_")[-1]
                    epoch = int(epoch_str)
                    epochs.append(epoch)
                except (IndexError, ValueError):
                    continue
            max_epoch = max(epochs) if epochs else None
            if max_epoch is not None:
                return cls.resolve_directory(query) / cls.file_name(max_epoch, "epoch")
            else:
                return None

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        query: "HecaNetwork.Query"
        base: BaseNetwork.Config
        feature_dim: int = 64
        eval_mode: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

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

    def upgrade(self, checkpoint):
        self.load_state_dict(checkpoint, strict=False)

    @classmethod
    def load(cls, query: "HecaNetwork.Query") -> "HecaNetwork":
        file = cls.File.get_latest(query)
        model = cls.search(query)
        if file:
            logger.info(f"Loading weights from: {file}")
            ckpt = torch.load(file, map_location=hardware.device)
            model.load_state_dict(ckpt["model_state"], strict=False)
        return model

    @classmethod
    def save(cls, query: "HecaNetwork.Query", tag: str, epoch: int) -> bool:
        path = cls.File.resolve_directory(query)
        file = cls.File.file_name(epoch, tag)
        torch.save({"model_state": cls.search(query).state_dict()}, path / file)
        logger.info(f"Saved model for epoch {epoch} with tag: {tag}.")
        return True
