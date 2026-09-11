from dataclasses import dataclass
import torch

from heca.learning.buffers.buffer import Buffer


class FairBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 2048
        gae_lambda: float = 0.95
        gamma: float = 0.95

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = [d.reward for d in self.queue]
        terminals = [d.terminal or d.truncated for d in self.queue]
        values = torch.stack([d.value for d in self.queue]).reshape(len(rewards))
        return self._gae_for_bucket(rewards, terminals, values)

    def _gae_for_bucket(self, rewards, terminals, values):
        T = len(rewards)  # Use the actual length of this group!

        advantages = torch.zeros(T)

        gae = 0.0
        next_value = 0.0 if terminals[-1] else values[-1]

        for t in reversed(range(T)):
            is_terminal = float(terminals[t])
            delta = (
                rewards[t] + self.cfg.gamma * next_value * (1 - is_terminal) - values[t]
            )
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - is_terminal) * gae
            advantages[t] = gae
            next_value = values[t]

        return advantages, advantages + values
