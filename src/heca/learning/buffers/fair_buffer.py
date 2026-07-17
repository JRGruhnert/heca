from dataclasses import dataclass
import torch

from heca.learning.buffers.buffer import Buffer, BufferData


class FairBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 2048
        gae_lambda: float = 0.95
        gamma: float = 0.99

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def is_allowed(self, data: BufferData) -> bool:
        raise NotImplementedError

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        adv_flat = torch.zeros(len(self.queue))
        ret_flat = torch.zeros(len(self.queue))

        for _, indices in self.tag_indices.items():
            group_items = [self.queue[i] for i in indices]
            rewards = [d.reward for d in group_items]
            terminals = [d.terminal or d.truncated for d in group_items]
            values = [d.value for d in group_items]
            advs, rets = self._gae_for_bucket(rewards, terminals, values)
            adv_flat[indices] = advs
            ret_flat[indices] = rets
        return adv_flat, ret_flat

    def _gae_for_bucket(self, rewards, terminals, values):
        T = len(rewards)  # Use the actual length of this group!

        advantages = torch.zeros(T)
        returns = torch.zeros(T)

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

        returns = advantages + torch.tensor(values)
        return advantages, returns
