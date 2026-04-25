from abc import abstractmethod
from dataclasses import dataclass
import torch
from torch import nn
from torch_geometric.data import Batch, HeteroData
from torch_geometric.nn import GINConv, GINEConv

from hoopgn.misc.mlp import GinStandardMLP, UnactivatedMLP
from hoopgn.networks.hoops.actors.actor import ActorReadoutNetwork
from hoopgn.networks.hoops.critics.critic import CriticReadoutNetwork
from hoopgn.networks.hoops.hoop import HoopNetwork


class V1Network(HoopNetwork):
    @dataclass(kw_only=True)
    class Query(HoopNetwork.Query):
        label: str = "gnn"

    @dataclass(kw_only=True)
    class Config(HoopNetwork.Config):
        actor: ActorReadoutNetwork.Config
        critic: CriticReadoutNetwork.Config
        feature_dim: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

        self.state_state_gin = GINConv(
            nn=GinStandardMLP(
                in_dim=self.cfg.feature_dim,
                out_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.feature_dim,
            ),
        )

        self.state_skill_gin = GINEConv(
            nn=GinStandardMLP(
                in_dim=self.cfg.feature_dim,
                out_dim=self.cfg.feature_dim,
                hidden_dim=self.cfg.feature_dim,
            ),
            edge_dim=2,
        )
        self.readout_net = GINConv(
            nn=UnactivatedMLP(self.cfg.feature_dim, 1),
        )

    def forward(self, x: HeteroData) -> tuple[torch.Tensor, torch.Tensor]:
        return self.actor(x), self.critic(x)

    def actor(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        raise NotImplementedError("Readout method must be implemented by subclass.")

    def critic(
        self, x: torch.Tensor, x_dict: dict, edge_index_dict: dict
    ) -> torch.Tensor:
        raise NotImplementedError("Readout method must be implemented by subclass.")

    def forward(self, x: HeteroData) -> torch.Tensor:
        if isinstance(x, Batch):
            x_dict = x.x_dict  # type: ignore
            edge_index_dict = x.edge_index_dict  # type: ignore
            edge_attr_dict = x.edge_attr_dict  # type: ignore
        elif len(args) == 2 and isinstance(args[0], dict) and isinstance(args[1], dict):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = self.edge_attr_dict
        elif (
            len(args) == 3
            and isinstance(args[0], dict)
            and isinstance(args[1], dict)
            and isinstance(args[2], dict)  # This is likely the mask from Captum
        ):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = args[2]
            # args[2] is the mask, which you may need to apply to x_dict or ignore
        elif len(args) == 3 and isinstance(args[0], dict) and isinstance(args[1], dict):
            x_dict = args[0]
            edge_index_dict = args[1]
            edge_attr_dict = self.edge_attr_dict
        else:
            raise ValueError(
                f"Invalid arguments {len(args)}. Expected (Batch), (x_dict, edge_index_dict), (x_dict, edge_index_dict, mask), or (x_dict, edge_index_dict, edge_attr_dict)."
            )

        x1 = self.state_state_gin(
            x=(x_dict["goal"], x_dict["obs"]),
            edge_index=edge_index_dict[("goal", "goal-obs", "obs")],
            # edge_attr=edge_attr_dict[("goal", "goal-obs", "obs")],
        )
        x2 = self.state_skill_gin(
            x=(x1, x_dict["task"]),
            edge_index=edge_index_dict[("obs", "obs-task", "task")],
            edge_attr=edge_attr_dict[("obs", "obs-task", "task")],
        )

        return self.readout(x2, x_dict, edge_index_dict)

    def explain(self, batch: Batch) -> tuple[HeteroExplanation, HeteroExplanation]:
        d: HeteroData = batch.get_example(0)  # type: ignore
        actor_explanation = self.actor_expl(
            d.x_dict,
            d.edge_index_dict,
            d.edge_attr_dict,
            index=torch.tensor([0]),
        )
        critic_explanation = self.critic_explainer(
            d.x_dict,
            d.edge_index_dict,
            d.edge_attr_dict,
            index=torch.tensor([0]),
        )
        assert isinstance(actor_explanation, HeteroExplanation)
        assert isinstance(critic_explanation, HeteroExplanation)
        return actor_explanation, critic_explanation

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
