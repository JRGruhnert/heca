from dataclasses import dataclass
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_

from heca.learning.buffers.stream_buffer import StreamBuffer
from heca.learning.learner import Learner
from heca.learning.buffers.buffer import Buffer


class APPO(Learner):
    @dataclass(kw_only=True)
    class Config(Learner.Config):
        folder: str = "appo"
        buffer: Buffer.Config = StreamBuffer.Config()
        # Hyperparameters
        lr: float = 0.0003
        max_grad_norm: float = 0.5
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_update: int = 1000
        eps_clip: float = 0.2

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def learn(self):
        assert isinstance(self.buffer, StreamBuffer)
        old_data = self.buffer.data
        old_actions = self.buffer.actions.detach().squeeze(-1)
        old_logprobs = self.buffer.logprobs.detach().squeeze(-1)
        logprobs, values, dist = self.network.evaluate(old_data, old_actions)
        entropy = dist.mean()

        adv, rtn = self.buffer.compute_advantages(logprobs, values)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        log_ratio = logprobs - old_logprobs.squeeze(-1).detach()
        ratios = torch.exp(log_ratio)

        surr1 = ratios * adv
        surr2 = torch.clamp(ratios, 1 - self.cfg.eps_clip, 1 + self.cfg.eps_clip) * adv
        policy_loss = -torch.min(surr1, surr2).mean()

        value_loss = self.mse_loss(values, rtn)
        loss = (
            policy_loss
            + self.cfg.critic_coef * value_loss
            - self.cfg.entropy_coef * entropy
        )

        self.optim.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network.parameters(), self.cfg.max_grad_norm)
        self.optim.step()
