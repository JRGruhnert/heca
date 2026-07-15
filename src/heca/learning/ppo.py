from dataclasses import dataclass
from functools import cached_property
import torch
import copy
from thop import profile
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.data import HeteroData

from heca.learning.buffers.appo_buffer import APPOBuffer
from heca.misc.hardware import device
from heca.misc.base import Configurable
from heca.heca_gnn.network import Network
from heca.learning.buffers.buffer import Buffer


class PPO(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str = "ppo"
        tag: str
        buffer: Buffer.Config
        network: Network.Config = Network.Config()
        # Hyperparameters
        batch_size: int = 64
        n_epoch: int = 5
        lr: float = 0.0003
        lr_annealing: bool = False
        eps_clip: float = 0.2
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: float | None = 0.02
        clip_value_loss: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.network: Network = Network.get(cfg.network)
        self.optim: torch.optim.Optimizer = torch.optim.AdamW(
            self.network.parameters(), lr=self.cfg.lr
        )

        self.buffer = Buffer.get(cfg.buffer)
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

    @cached_property
    def inference_network(self) -> Network:
        """All agents attached to this PPO share this frozen copy."""
        return copy.deepcopy(self.network)

    def sync_inference(self):
        """Push training weights into the shared inference copy."""
        self.inference_network.load_state_dict(self.network.state_dict())

    def learn(self, progress: float):
        assert self.buffer.full

        if isinstance(self.buffer, APPOBuffer):
            # V-trace needs current policy evaluation first
            all_data = self.buffer.flat_data
            all_actions = self.buffer.flat_actions.detach().squeeze(-1)
            current_logprobs, current_values, dist = self.network.evaluate(
                all_data, all_actions
            )
            current_logprobs = current_logprobs.squeeze(-1)
            current_values = current_values.squeeze(-1)
            current_entropy = dist.entropy().mean()

            adv, rtn = self.buffer.compute_advantages(
                current_logprobs=current_logprobs,
                current_values=current_values,
            )
            self._mini_batch_loop(
                adv,
                rtn,
                cached_logprobs=current_logprobs.detach(),
                cached_values=current_values.detach(),
                cached_entropy=current_entropy,
            )
        else:
            # GAE for PPO and DDPPO
            adv, rtn = self.buffer.compute_advantages()
            self._mini_batch_loop(adv, rtn)
            if self.cfg.lr_annealing:
                lr = self.cfg.lr * (1.0 - progress)
                for pg in self.optim.param_groups:
                    pg["lr"] = lr
        self.sync_inference()

    def _mini_batch_loop(
        self,
        adv: torch.Tensor,
        rtn: torch.Tensor,
        cached_logprobs=None,
        cached_values=None,
        cached_entropy=None,
    ):
        B = self.buffer.cfg.capacity

        old_data = self.buffer.flat_data
        old_actions = self.buffer.flat_actions.detach().squeeze(-1)
        old_logprobs = self.buffer.flat_logprobs.detach().squeeze(-1)
        old_values = self.buffer.flat_values.detach().squeeze(-1)

        kl_stop = False
        for _ in range(self.cfg.n_epoch):
            indices = torch.randperm(B)
            for start in range(0, B, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_idx = indices[start:end]
                mb_ilist = mb_idx.tolist()

                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]
                mb_old_val = old_values[mb_idx]

                if (
                    cached_logprobs is not None
                    and cached_values is not None
                    and cached_entropy is not None
                ):
                    # ── APPO: use cached forward pass ──
                    logprobs = cached_logprobs[mb_idx]
                    state_values = cached_values[mb_idx]
                    entropy = cached_entropy
                else:
                    # ── PPO/DDPPO: fresh forward pass per minibatch ──
                    mb_data = [old_data[i] for i in mb_ilist]
                    logprobs, state_values, dist = self.network.evaluate(
                        mb_data, mb_actions
                    )
                    logprobs = logprobs.squeeze(-1)
                    state_values = state_values.squeeze(-1)
                    entropy = dist.entropy().mean()

                assert isinstance(entropy, torch.Tensor)
                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO ratio
                ratios = torch.exp(logprobs - mb_logprobs.detach().to(device))

                # Policy loss
                surr1 = ratios * mb_adv
                surr2 = (
                    torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip)
                    * mb_adv
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                if self.cfg.clip_value_loss:
                    values_pred = mb_old_val + torch.clamp(
                        state_values - mb_old_val,
                        -self.cfg.eps_clip,
                        self.cfg.eps_clip,
                    )
                    value_loss = self.mse_loss(values_pred, mb_rtn)
                else:
                    value_loss = self.mse_loss(state_values, mb_rtn)

                # Entropy bonus
                loss = (
                    policy_loss
                    + self.cfg.critic_coef * value_loss
                    - self.cfg.entropy_coef * entropy
                )

                # Gradient step
                self.optim.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
                self.optim.step()

                # KL early stopping
                with torch.no_grad():
                    log_ratio = logprobs - mb_logprobs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    if (
                        self.cfg.target_kl is not None
                        and approx_kl > self.cfg.target_kl
                    ):
                        kl_stop = True
                        break
            if kl_stop:
                break

    def metadata(self) -> dict:
        return {
            "mini_batch_size": self.cfg.batch_size,
            "learning_epochs": self.cfg.n_epoch,
            "lr_annealing": self.cfg.lr_annealing,
            "learning_rate": self.cfg.lr,
            "eps_clip": self.cfg.eps_clip,
            "entropy_coef": self.cfg.entropy_coef,
            "critic_coef": self.cfg.critic_coef,
            "max_grad_norm": self.cfg.max_grad_norm,
            "clip_value_loss": self.cfg.clip_value_loss,
        } | (
            {"target_kl": self.cfg.target_kl} if self.cfg.target_kl is not None else {}
        )

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
            self.buffer.store_prediction(data, action, logprob, value, tag)
        else:
            with torch.inference_mode():
                logits = self.network.actor(data)
            action = logits.argmax(dim=-1)
        return int(action)

    def update(self, reward: float, terminal: bool, truncated: bool, tag: str):
        if self.train_mode:
            self.buffer.store_feedback(reward, terminal, truncated, tag)
