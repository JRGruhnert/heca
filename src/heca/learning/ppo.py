from dataclasses import dataclass
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_


from heca.learning.buffers.fair_buffer import FairBuffer
from heca.learning.learner import Learner
from heca.misc.hardware import device
from heca.learning.buffers.buffer import Buffer


class PPO(Learner):
    @dataclass(kw_only=True)
    class Config(Learner.Config):
        folder: str = "ppo"
        buffer: Buffer.Config = FairBuffer.Config()
        # Hyperparameters
        batch_size: int = 64
        n_epoch: int = 10
        lr: float = 0.0003
        lr_annealing: bool = False
        eps_clip: float = 0.2
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: float | None = 0.02
        clip_value_loss: bool = True
        max_update: int = 1000

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def learn(self):
        adv, rtn = self.buffer.compute_advantages()
        # Count how many steps have a meaningful absolute advantage > 0.01
        meaningful_steps = (adv.abs() > 0.01).sum().item()
        print(f"Meaningful gradient steps: {meaningful_steps} / {len(adv)}")
        self._mini_batch_loop(adv, rtn)
        if self.cfg.lr_annealing:
            lr = self.cfg.lr * (1.0 - self.current_update / self.cfg.max_update)
            for pg in self.optim.param_groups:
                pg["lr"] = lr

    def _mini_batch_loop(self, adv: torch.Tensor, rtn: torch.Tensor):
        old_data = self.buffer.data
        old_actions = self.buffer.actions.detach().squeeze(-1)
        old_logprobs = self.buffer.logprobs.detach().squeeze(-1)
        old_values = self.buffer.values.detach().squeeze(-1)

        kl_stop = False
        for _ in range(self.cfg.n_epoch):
            indices = torch.randperm(self.buffer.cfg.capacity)
            for start in range(0, self.buffer.cfg.capacity, self.cfg.batch_size):
                end = start + self.cfg.batch_size
                mb_idx = indices[start:end]
                mb_ilist = mb_idx.tolist()

                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]
                mb_old_val = old_values[mb_idx]
                mb_data = [old_data[i] for i in mb_ilist]
                logprobs, state_values, dist = self.network.evaluate(
                    mb_data, mb_actions
                )
                entropy = dist.mean()

                assert isinstance(entropy, torch.Tensor)
                # Normalize advantages
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # PPO ratio
                ratios = torch.exp(logprobs - mb_logprobs)

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
