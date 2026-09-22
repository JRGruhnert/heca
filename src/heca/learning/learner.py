from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
import torch
import copy
import wandb
from wandb.wandb_run import Run
from thop import profile
from torch import nn
from torch.distributions import Categorical
from torch_geometric.explain import Explainer, CaptumExplainer

from heca.learning.buffers.fair_buffer import FairBuffer
from heca.learning.reward_normalizer import RewardNormalizer
from heca.misc import hardware, logger
from heca.misc.base import Persistable, latest_checkpoint
from heca.data.entity import Entity
from heca.graphs.data import HecaData, TrunkMemory
from heca.heca_gnn.network import Network, NetworkOutput
from heca.learning.buffers.buffer import Buffer, BufferData
from heca.scenes.scene import SceneFeedback

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


def _stored(memory: TrunkMemory) -> TrunkMemory:
    return {name: value.detach().clone() for name, value in memory.items()}


@dataclass(slots=True)
class TempStore:
    data: HecaData
    action: torch.Tensor
    logprob: torch.Tensor
    value: torch.Tensor

    def complete(self, fb: SceneFeedback) -> BufferData:
        return BufferData(
            data=self.data,
            action=self.action,
            logprob=self.logprob,
            value=self.value,
            reward=fb.reward,
            terminal=fb.terminal,
            truncated=fb.truncated,
        )


class Learner(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "network"
        label: str = "standard"
        buffer: Buffer.Config = FairBuffer.Config()
        network: Network.Config
        wandb: logger.WandBConfig = logger.WandBConfig()
        # Hyperparameters
        lr: float
        max_grad_norm: float
        entropy_coef: float
        critic_coef: float
        eps_clip: float
        lr_annealing: bool
        max_update: int
        # Additional Training Hyperparameters
        normalize_rewards: bool = False
        # Misc
        save_interval: int = 50
        group: str = ""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.network = Network.get(cfg.network)
        self.optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.cfg.lr
        )
        self.metrics: dict[str, float] = {}
        self.current_update: int = 0
        self.normalizer: RewardNormalizer = RewardNormalizer()
        self.buffer = Buffer.get(cfg.buffer)
        self.pocket: TempStore | None = None
        self.train_mode = True

        self._mem_next: TrunkMemory = {}

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

    @cached_property
    def inference_net(self) -> Network:
        """All agents attached to this PPO share this frozen copy."""
        return copy.deepcopy(self.network)

    def _sync_inference(self):
        """Push training weights into the shared inference copy."""
        self.inference_net.load_state_dict(self.network.state_dict())

    def sync(self):
        """Post-update synchronization hook. Overridden by FPPO."""
        self._sync_inference()
        self._periodic_save()

    def _periodic_save(self):
        interval = self.cfg.save_interval
        if interval > 0 and self.current_update % interval == 0:
            self.save()

    def learn(self):
        raise NotImplementedError

    def measure_flops(self, data: HecaData) -> tuple[int, int]:
        assert self.network is not None
        with torch.no_grad():
            result = profile(self.network, inputs=data, verbose=False)
            return int(result[0]), int(result[1])

    def eval(self):
        self.train_mode = False

    def predict(self, data: HecaData) -> int:
        data.memory = self._mem_next
        self._mem_next = {}

        if self.train_mode:
            net = self.inference_net
            with torch.inference_mode():
                out: NetworkOutput = net(data)
            dist = Categorical(logits=out.logits)
            action = dist.sample()
            self.pocket = TempStore(
                data=data,
                action=action,
                logprob=dist.log_prob(action),
                value=out.value,
            )
        else:
            with torch.inference_mode():
                out = self.network(data)
            action = out.logits.argmax(dim=-1)
        if self.network.cfg.use_memory:
            self._mem_next = _stored(out.memory)
        return int(action)

    def _init_wandb(self):
        self._wandb_run: Run | None = None
        if not self.cfg.wandb.enabled:
            return

        config_dict = {
            "lr": self.cfg.lr,
            "max_grad_norm": self.cfg.max_grad_norm,
            "entropy_coef": self.cfg.entropy_coef,
            "critic_coef": self.cfg.critic_coef,
            "eps_clip": self.cfg.eps_clip,
            "normalize_rewards": self.cfg.normalize_rewards,
            # Buffer config
            "buffer/capacity": self.cfg.buffer.capacity,
            "buffer/label": str(type(self.cfg.buffer)),
            # Network config
            "network/input_dim": Entity.FEATURE_DIM,
            "network/max_state": Entity.MAX_STATE_DIM,
            "network/use_option_interaction": (self.cfg.network.use_option_transformer),
            "network/use_timeline_memory": (self.cfg.network.use_memory),
        }

        self._wandb_run = wandb.init(
            project=self.cfg.wandb.project,
            entity=self.cfg.wandb.entity,
            name=self.cfg.tag,
            group=self.cfg.group or None,
            config=config_dict,
            mode=self.cfg.wandb.mode,
            save_code=self.cfg.wandb.save_code,
            tags=[self.cfg.label],
            reinit="create_new",
        )

        wandb.run = self._wandb_run

        if self.cfg.wandb.watch_model:
            self._wandb_run.watch(
                self.network,
                log="gradients",
                log_freq=self.cfg.wandb.watch_freq,
                log_graph=True,
            )

    @property
    def run(self) -> Run | None:
        return self._wandb_run

    def finish(self, exit_code: int | None = None):
        if self._wandb_run is None:
            return
        self._wandb_run.finish(exit_code=exit_code)

    def training_log(self):
        display_metrics = {k.removeprefix("train/"): v for k, v in self.metrics.items()}
        metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in display_metrics.items()])
        logger.info(f"Update {self.current_update:4d} | {metrics_str}")

        if self.cfg.wandb.enabled:
            self._wandb_run.log(
                {k: v for k, v in self.metrics.items()},
            )

    def update(self, fb: SceneFeedback) -> bool:
        if self.cfg.normalize_rewards:
            fb.reward = self.normalizer.update(fb.reward)
        if self.train_mode:
            assert isinstance(self.pocket, TempStore)
            data = self.pocket.complete(fb)
            if self.buffer.add(data):
                self.learn()
                self.current_update += 1
                self.metrics.update(self.buffer.stats())
                self.training_log()
                self.buffer.reset()
                return True
        if fb.end:
            self._mem_next = {}
        return False

    def _checkpoint(self) -> dict:
        return {
            "network": self.network.state_dict(),
            "optimizer": self.optim.state_dict(),
            "current_update": self.current_update,
            "reward_normalizer": {
                "mean": self.normalizer.mean,
                "var": self.normalizer.var,
                "count": self.normalizer.count,
            },
        }

    def _restore(self, checkpoint: dict):
        self.network.load_state_dict(checkpoint["network"])
        self.optim.load_state_dict(checkpoint["optimizer"])
        self.current_update = int(checkpoint["current_update"])

        # Restore per-tag normalizers
        if "reward_normalizer" in checkpoint:
            self.normalizer.mean = checkpoint["reward_normalizer"]["mean"]
            self.normalizer.var = checkpoint["reward_normalizer"]["var"]
            self.normalizer.count = checkpoint["reward_normalizer"]["count"]
            logger.info(f"Restored normalizer")

        self._sync_inference()

    def _save(self, path: Path):
        filepath = path / f"ckp_{self.current_update}.pt"
        torch.save(self._checkpoint(), filepath)
        logger.info(f"Saved full checkpoint to {filepath}")

    def _load(self, path: Path):
        filepath = latest_checkpoint(path, "ckp")
        if filepath is None:
            logger.warning(f"No checkpoint found at {path}. Starting from scratch.")
            return

        checkpoint = torch.load(
            filepath, map_location=hardware.device, weights_only=False
        )
        if "current_update" not in checkpoint:
            # checkpoints written before the payload carried it
            checkpoint["current_update"] = int(filepath.stem.rsplit("_", 1)[-1])

        self._restore(checkpoint)
        logger.info(
            f"Loaded full checkpoint from {filepath} at update {self.current_update}"
        )
