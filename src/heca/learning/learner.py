from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Literal
import torch
import copy
import wandb
from thop import profile
from torch import nn
from torch.distributions import Categorical
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.data import HeteroData

from heca.learning.reward_normalizer import RewardNormalizer
from heca.misc import hardware, logger
from heca.misc.base import Persistable
from heca.heca_gnn.network import Network
from heca.learning.buffers.buffer import Buffer, BufferData


@dataclass(kw_only=True, slots=True)
class WandBConfig:
    project: str = "master-thesis"
    entity: str = "heca-university-freiburg"
    mode: Literal["online", "offline", "disabled"] = "online"
    save_code: bool = False  # Uploads training script
    watch_model: bool = True  # Log gradients & weight histograms
    watch_freq: int = 100  # Frequency of gradient logging
    enabled: bool = False


# class _ExplainerWrapper(nn.Module):
#     def __init__(self, model: nn.Module):
#         super().__init__()
#         self.model = model
#         self._edge_attr_dict: dict = {}

#     def set_edge_attrs(self, edge_attr_dict: dict):
#         self._edge_attr_dict = edge_attr_dict

#     def forward(self, x_dict: dict, edge_index_dict: dict):
#         data = HeteroData()
#         for key, x in x_dict.items():
#             data[key].x = x
#         for key, edge_index in edge_index_dict.items():
#             data[key].edge_index = edge_index
#         for key, edge_attr in self._edge_attr_dict.items():
#             data[key].edge_attr = edge_attr
#         batch = Batch.from_data_list([data])
#         return self.model(batch)


class Learner(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "learner"
        buffer: Buffer.Config
        network: Network.Config = Network.Config()
        # Hyperparameters
        lr: float
        max_grad_norm: float
        entropy_coef: float
        critic_coef: float
        max_update: int
        eps_clip: float
        normalize_rewards: bool
        wandb: WandBConfig = WandBConfig()

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.network: Network = Network.get(cfg.network)
        self.optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.cfg.lr
        )
        self.metrics: dict[str, float] = {}
        self.normalizers: dict[str, RewardNormalizer] = {}
        self.buffer = Buffer.get(cfg.buffer)
        self.pocket: dict[str, BufferData] = {}
        self.train_mode = True
        self._init_wandb()
        self.explainer = Explainer(
            self.network,
            algorithm=CaptumExplainer("Saliency"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

        self.n_terminals = 0
        self.n_truncated = 0
        self.n_feedback = 0
        self.n_episodes = 0
        # self.actor_explainer = Explainer(
        #     _ExplainerWrapper(self.actor),
        #     algorithm=CaptumExplainer("IntegratedGradients"),
        #     explanation_type="phenomenon",
        #     node_mask_type="attributes",
        #     edge_mask_type=None,
        #     model_config=dict(
        #         mode="multiclass_classification",
        #         task_level="graph",
        #         return_type="raw",
        #     ),
        # )

        # self.critic_explainer = Explainer(
        #     _ExplainerWrapper(self.critic),
        #     algorithm=CaptumExplainer("IntegratedGradients"),
        #     explanation_type="model",
        #     node_mask_type="attributes",
        #     edge_mask_type=None,
        #     model_config=dict(
        #         mode="regression",
        #         task_level="graph",
        #         return_type="raw",
        #     ),
        # )
        self.current_update = 0

    def register(self, tag: str) -> "Learner":
        if tag not in self.normalizers:
            self.normalizers[tag] = RewardNormalizer()
        self.buffer.tags.add(tag)
        return self

    @cached_property
    def inference_network(self) -> Network:
        """All agents attached to this PPO share this frozen copy."""
        return copy.deepcopy(self.network)

    def sync_inference(self):
        """Push training weights into the shared inference copy."""
        self.inference_network.load_state_dict(self.network.state_dict())

    def learn(self):
        raise NotImplementedError

    def measure_flops(self, data: HeteroData) -> tuple[int, int]:
        assert self.network is not None
        with torch.no_grad():
            result = profile(self.network, inputs=data, verbose=False)
            return int(result[0]), int(result[1])

    def eval(self):
        self.train_mode = False

    # def explain(
    #     self,
    #     obs: StateValueDict,
    #     goal: StateValueDict,
    # ) -> tuple[
    #     torch.Tensor,
    #     torch.Tensor,
    #     torch.Tensor,
    #     HeteroExplanation,
    #     HeteroExplanation,
    # ]:
    #     # Resolve the action first (same logic as act())
    #     action, logprob, value = self.act(obs, goal)

    #     # Build the graph for the current observation
    #     data = self.to_data(obs, goal)

    #     # Edge attributes are not perturbed by the explainer; store them in the wrappers
    #     self.actor_explainer.model.set_edge_attrs(data.edge_attr_dict)
    #     self.critic_explainer.model.set_edge_attrs(data.edge_attr_dict)

    #     actor_x = {k: v for k, v in data.x_dict.items() if k != "critic"}
    #     actor_ei = {
    #         k: v
    #         for k, v in data.edge_index_dict.items()
    #         if k[0] != "critic" and k[2] != "critic"
    #     }

    #     actor_explanation = self.actor_explainer(
    #         actor_x,
    #         actor_ei,
    #         target=torch.tensor([int(action.item())]),
    #     )

    #     critic_x = {k: v for k, v in data.x_dict.items() if k != "actor"}
    #     critic_ei = {
    #         k: v
    #         for k, v in data.edge_index_dict.items()
    #         if k[0] != "actor" and k[2] != "actor"
    #     }

    #     critic_explanation = self.critic_explainer(
    #         critic_x,
    #         critic_ei,
    #     )

    #     return action, logprob, value, actor_explanation, critic_explanation

    def predict(self, data: HeteroData, tag: str) -> int:
        if self.train_mode:
            net = self.inference_network
            with torch.inference_mode():
                logits, value = net.forward(data)
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob = dist.log_prob(action)
            self.pocket[tag] = BufferData(
                tag=tag,
                data=data,
                action=action,
                logprob=logprob,
                value=value,
            )
        else:
            with torch.inference_mode():
                logits = self.network.actor(data)
            action = logits.argmax(dim=-1)
        return int(action)

    def _init_wandb(self):
        if not self.cfg.wandb.enabled:
            return
        config_dict = {
            "lr": self.cfg.lr,
            "max_grad_norm": self.cfg.max_grad_norm,
            "entropy_coef": self.cfg.entropy_coef,
            "critic_coef": self.cfg.critic_coef,
            "max_update": self.cfg.max_update,
            "eps_clip": self.cfg.eps_clip,
            "normalize_rewards": self.cfg.normalize_rewards,
            # Buffer config
            "buffer/capacity": self.cfg.buffer.capacity,
            "buffer/label": str(type(self.cfg.buffer)),
            # Network config
            "network/input_dim": self.cfg.network.input_feat_dim,
            "network/feature_dim": self.cfg.network.feature_dim,
            "network/num_stepmix_layers": self.cfg.network.num_stepmix_layers,
            "network/num_tapas_layers": self.cfg.network.num_tapas_layers,
        }

        self._wandb_run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.tag,
            config=config_dict,
            mode=self.cfg.wandb.mode,
            save_code=self.cfg.wandb.save_code,
            tags=[self.cfg.label, self.cfg.label],
        )

        if self.cfg.wandb.watch_model:
            wandb.watch(
                self.network,
                log="gradients",
                log_freq=self.cfg.wandb.watch_freq,
                log_graph=True,
            )

    def training_log(self):
        display_metrics = {k.removeprefix("train/"): v for k, v in self.metrics.items()}
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in display_metrics.items()])
        logger.info(f"Update {self.current_update:4d} | {metrics_str}")

        if self.cfg.wandb.enabled:
            wandb.log(self.metrics, step=self.current_update)

    def update(self, reward: float, terminal: bool, truncated: bool, tag: str) -> bool:
        if self.cfg.normalize_rewards:
            reward = self.normalizers[tag].update(reward)
        if self.train_mode:
            data = self.pocket[tag]
            data.reward = reward
            data.terminal = terminal
            data.truncated = truncated
            if self.buffer.add(data):
                self.learn()
                self.current_update = min(self.current_update + 1, self.cfg.max_update)
                self.sync_inference()
                self.metrics.update(self.buffer.stats())
                self.training_log()
                self.buffer.reset()

                if self.current_update % 100 == 0:
                    self.save()

        return self.current_update >= self.cfg.max_update

    def _save(self, path: Path):
        filepath = path / f"checkpoint_{self.current_update}.pt"

        checkpoint = {
            "network": self.network.state_dict(),
            "optimizer": self.optim.state_dict(),
            "reward_normalizers": {
                tag: {
                    "mean": norm.mean,
                    "var": norm.var,
                    "count": norm.count,
                }
                for tag, norm in self.normalizers.items()
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved full checkpoint to {filepath}")

    def _load(self, path: Path):
        filepath = path / "checkpoint.pt"
        if not filepath.exists():
            logger.warning(f"No checkpoint found at {filepath}. Starting from scratch.")
            return

        checkpoint = torch.load(filepath, map_location=hardware.device)

        self.network.load_state_dict(checkpoint["network"])
        self.optim.load_state_dict(checkpoint["optimizer"])

        # Restore per-tag normalizers
        if "reward_normalizers" in checkpoint:
            for tag, stats in checkpoint["reward_normalizers"].items():
                if tag not in self.normalizers:
                    self.normalizers[tag] = RewardNormalizer()
                norm = self.normalizers[tag]
                norm.mean = stats["mean"]
                norm.var = stats["var"]
                norm.count = stats["count"]
            logger.info(f"Restored {len(self.normalizers)} per-tag normalizers")

        self.sync_inference()
        logger.info(
            f"Loaded full checkpoint from {filepath} at update {self.current_update}"
        )
