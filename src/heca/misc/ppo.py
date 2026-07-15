# heca/misc/ppo.py (rewritten)
from dataclasses import dataclass
from functools import cached_property
import torch
import copy
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from heca.misc.base import Configurable
from heca.heca_gnn.network import Network
from heca.misc.buffer import RolloutBuffer
from heca.misc.hardware import device
from thop import profile
from torch_geometric.data import HeteroData


class PPO(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str = "ppo"
        tag: str = "default"
        # Hyperparameters only — no buffer or network config here
        mini_batch_size: int = 64
        learning_epochs: int = 5
        lr: float = 0.0003
        lr_annealing: bool = False
        gamma: float = 0.99
        gae_lambda: float = 0.95
        eps_clip: float = 0.2
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: float | None = 0.02
        clip_value_loss: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()
        self.network: Network | None = None
        self.optim: torch.optim.Optimizer | None = None

    def setup(self, config: Network.Config) -> "PPO":
        """Create network and optimizer. Idempotent — safe to call multiple times."""
        if self.network is not None:
            return self
        self.network = Network.get(config)
        self.optim = torch.optim.AdamW(self.network.parameters(), lr=self.cfg.lr)
        return self

    @cached_property
    def inference_network(self) -> Network:
        """All agents attached to this PPO share this frozen copy."""
        assert self.network is not None
        return copy.deepcopy(self.network)

    def sync_inference(self):
        """Push training weights into the shared inference copy."""
        assert self.network is not None
        self.inference_network.load_state_dict(self.network.state_dict())

    def copy_network(self) -> Network:
        assert self.network is not None
        return copy.deepcopy(self.network)

    def learn(self, buffer: RolloutBuffer, progress: float) -> tuple[dict, bool]:
        """Run PPO update on the provided buffer.

        Args:
            buffer: Filled RolloutBuffer.
            progress: Training progress 0..1 for LR annealing.
        Returns:
            (state_dict, is_best) — training network state and whether
            the success rate is a new high score.
        """
        assert self.network is not None
        assert self.optim is not None
        assert buffer.full, "Buffer must be full before learning"

        self._mini_batch_loop(buffer)

        # LR annealing
        if self.cfg.lr_annealing:
            lr = self.cfg.lr * (1.0 - progress)
            for pg in self.optim.param_groups:
                pg["lr"] = lr

        return self.network.state_dict(), buffer.flush_and_rate()

    def _mini_batch_loop(self, buffer: RolloutBuffer):
        assert self.network is not None
        assert self.optim is not None

        adv, rtn = buffer.compute_gae()
        B = len(buffer)
        old_data = buffer.data
        old_actions = torch.stack(buffer.actions, dim=0).detach().to(device).squeeze(-1)
        old_logprobs = (
            torch.stack(buffer.logprobs, dim=0).detach().to(device).squeeze(-1)
        )
        old_values = torch.stack(buffer.values, dim=0).detach().to(device).squeeze(-1)

        kl_stop = False
        for epoch in range(self.cfg.learning_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = indices[start:end]
                mb_ilist = mb_idx.tolist()

                mb_data = [old_data[i] for i in mb_ilist]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]
                mb_old_val = old_values[mb_idx]

                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Forward pass
                logprobs, state_values, dist = self.network.evaluate(
                    mb_data, mb_actions
                )
                state_values = state_values.squeeze(-1)

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
                entropy = dist.entropy().mean()
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
            "mini_batch_size": self.cfg.mini_batch_size,
            "learning_epochs": self.cfg.learning_epochs,
            "lr_annealing": self.cfg.lr_annealing,
            "learning_rate": self.cfg.lr,
            "gamma": self.cfg.gamma,
            "gae_lambda": self.cfg.gae_lambda,
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
