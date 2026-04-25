from dataclasses import dataclass
from pathlib import Path
from tensordict import TensorDict
import torch
from torch import nn
from torch_geometric.data import HeteroData

from hoopgn.entities.properties.encoders.encoder import PropertyEncoder
from hoopgn.misc import hardware, logger
from hoopgn.misc.classes import StoragableClass
from hoopgn.misc.td import TDScene
from hoopgn.networks.hoops.actors.actor import ActorReadoutNetwork
from hoopgn.networks.hoops.critics.critic import CriticReadoutNetwork
from torch_geometric.data import HeteroData


class HoopNetwork(StoragableClass, nn.Module):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        actor: ActorReadoutNetwork.Config
        critic: CriticReadoutNetwork.Config
        feature_dim: int = 32
        eval_mode: bool = False

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

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

    def from_disk(self):
        path = self.path / "checkpoint_epoch0.pt"
        checkpoint = torch.load(path, map_location=hardware.device)
        self.load_state_dict(checkpoint["model_state"], strict=False)

    def to_disk(self, highscore: bool, index: int):
        if highscore:
            tag = Path("highscore_epoch{}.pt".format(index))
        else:
            tag = Path("checkpoint_epoch{}.pt".format(index))

        logger.info(f"Saving weights to: {self.path / tag} for epoch {index}")
        torch.save({"state": self.state_dict()}, self.path / tag)
