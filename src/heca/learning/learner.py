from dataclasses import dataclass
from functools import cached_property
import torch
import copy
from thop import profile
from torch import nn
from torch.distributions import Categorical
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.data import HeteroData

from heca.learning.reward_normalizer import RewardNormalizer
from heca.misc.base import Persistable
from heca.heca_gnn.network import Network
from heca.learning.buffers.buffer import Buffer, BufferData


class Learner(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        label: str = "learner"
        buffer: Buffer.Config
        network: Network.Config = Network.Config()
        # Hyperparameters
        lr: float
        max_grad_norm: float
        entropy_coef: float
        critic_coef: float
        max_update: int
        eps_clip: float

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.network: Network = Network.get(cfg.network)
        self.optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.cfg.lr
        )
        self.reward_normalizer = RewardNormalizer()
        self.buffer = Buffer.get(cfg.buffer)
        self.pocket: dict[str, BufferData] = {}
        self.train_mode = True

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
        self.current_update = 0

    def register(self, tag: str) -> "Learner":
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

    def explain(self, data: HeteroData):
        self.explainer.get_prediction()

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

    def update(self, reward: float, terminal: bool, truncated: bool, tag: str) -> bool:
        normalized_reward = self.reward_normalizer.update(reward)
        if self.train_mode:
            data = self.pocket[tag]
            data.reward = reward
            data.terminal = terminal
            data.truncated = truncated
            if self.buffer.add(data):
                self.learn()
                self.current_update = min(self.current_update + 1, self.cfg.max_update)
                self.sync_inference()
                self.buffer.reset()

        return self.current_update >= self.cfg.max_update
