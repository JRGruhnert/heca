from dataclasses import dataclass
from functools import cached_property
import torch
import copy
from thop import profile
from enum import Enum
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch_geometric.explain import Explainer, CaptumExplainer

from heca.misc.hardware import device
from heca.misc.base import Configurable
from heca.heca_gnn.network import Network
from torch_geometric.data import HeteroData
from torch.distributions import Categorical
from heca.learning.buffers.buffer import Buffer


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class PPO(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str = "ppo"
        tag: str
        buffer: Buffer.Config
        network: Network.Config = Network.Config()
        # Hyperparameters
        mini_batch_size: int = 64
        learning_epochs: int = 5
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
        assert self.network is not None
        return copy.deepcopy(self.network)

    def sync_inference(self):
        """Push training weights into the shared inference copy."""
        assert self.network is not None
        self.inference_network.load_state_dict(self.network.state_dict())

    def learn2(self, progress: float) -> tuple[dict, bool]:
        assert self.network is not None
        assert self.optim is not None
        assert self.buffer.full, "Buffer must be full before learning"

        # --- APPO Mode ---
        # 1. Get all data from buffer (flattened)
        all_data = self.buffer.flat_data
        all_actions = self.buffer.flat_actions.detach().squeeze(-1)

        # 2. Forward pass on the ENTIRE buffer (current policy π)
        with torch.set_grad_enabled(True):
            # We need gradients, so no inference_mode
            current_logprobs, current_values, _ = self.network.evaluate(
                all_data, all_actions
            )
            current_logprobs = current_logprobs.squeeze(-1)
            current_values = current_values.squeeze(-1)

        # 3. Compute V-trace targets (uses current logprobs and values)
        adv, rtn = self.buffer.compute_vtrace(current_logprobs, current_values)

        # 4. Now run the minibatch loop using these fixed targets
        B = len(self.buffer)
        old_logprobs = self.buffer.flat_logprobs.detach().squeeze(-1)
        old_values = self.buffer.flat_values.detach().squeeze(-1)

        kl_stop = False
        for epoch in range(self.cfg.learning_epochs):
            indices = torch.randperm(B)
            for start in range(0, B, self.cfg.mini_batch_size):
                mb_idx = indices[start : start + self.cfg.mini_batch_size]
                mb_ilist = mb_idx.tolist()

                mb_data = [all_data[i] for i in mb_ilist]
                mb_actions = all_actions[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]

                # Recompute logprobs and values for the minibatch (current policy)
                # We could use the cached `current_logprobs[mb_idx]` here,
                # but recomputing is standard and ensures gradients flow.
                # To save compute, we can detach and use the cached ones.
                # Let's use the cached ones to save time:
                mb_logprobs = current_logprobs[mb_idx].detach()  # π
                mb_old_logprobs = old_logprobs[mb_idx]  # μ
                mb_values = current_values[mb_idx].detach()
                mb_old_values = old_values[mb_idx]

                # PPO ratio = π / μ
                ratios = torch.exp(mb_logprobs - mb_old_logprobs)

                # Normalize advantages
                mb_adv_norm = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Policy loss (clipped)
                surr1 = ratios * mb_adv_norm
                surr2 = (
                    torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip)
                    * mb_adv_norm
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss (clipped)
                if self.cfg.clip_value_loss:
                    values_pred = mb_old_values + torch.clamp(
                        mb_values - mb_old_values,
                        -self.cfg.eps_clip,
                        self.cfg.eps_clip,
                    )
                    value_loss = self.mse_loss(values_pred, mb_rtn)
                else:
                    value_loss = self.mse_loss(mb_values, mb_rtn)

                # Entropy (we need the distribution for entropy)
                # We didn't store dist in forward pass, let's recompute entropy quickly:
                # Actually, we can compute entropy from logits, but let's get dist in forward pass.
                # Let's modify the forward pass to return logits.

                # ... Gradient step ...

        # CRITICAL: DO NOT clear the buffer! FIFO eviction handles memory.
        return self.network.state_dict(), False  # No highscore clearing in APPO

    def learn(self, progress: float) -> tuple[dict, bool]:
        assert self.network is not None
        assert self.optim is not None
        assert self.buffer.full, "Buffer must be full before learning"

        self._mini_batch_loop()

        # LR annealing
        if self.cfg.lr_annealing:
            lr = self.cfg.lr * (1.0 - progress)
            for pg in self.optim.param_groups:
                pg["lr"] = lr

        self.sync_inference()

        return self.network.state_dict(), self.buffer.flush_and_rate()

    def _mini_batch_loop(self):
        assert self.network is not None
        assert self.optim is not None

        adv, rtn = self.buffer.compute_gae()
        B = len(self.buffer)
        assert B == self.buffer.capacity  # sanity check

        old_data = self.buffer.flat_data
        old_actions = self.buffer.flat_actions.detach().squeeze(-1)
        old_logprobs = self.buffer.flat_logprobs.detach().squeeze(-1)
        old_values = self.buffer.flat_values.detach().squeeze(-1)

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
