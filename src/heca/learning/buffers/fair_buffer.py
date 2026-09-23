from dataclasses import dataclass
import torch

from heca.learning.buffers.buffer import Buffer


class FairBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 2048
        gae_lambda: float = 0.95
        gamma: float = 0.99

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def compute_advantages(self) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = [d.reward for d in self.queue]
        terminals = [d.terminal for d in self.queue]
        truncates = [d.truncated for d in self.queue]
        values = torch.stack([d.value for d in self.queue]).reshape(len(rewards))
        return self._gae_for_bucket(rewards, terminals, truncates, values)

    def _gae_for_bucket(self, rewards, terminals, truncates, values):
        T = len(rewards)  # Use the actual length of this group!

        advantages = torch.zeros(T, device=values.device, dtype=values.dtype)

        gae = 0.0
        next_value = 0.0 if terminals[-1] else values[-1]

        for t in reversed(range(T)):
            is_terminal = bool(terminals[t])
            is_truncated = bool(truncates[t])

            # Value of the state *after* this transition.
            #   terminal  -> 0 (the episode truly ended)
            #   truncated -> V(s_t) itself: V(s_{t+1}) is never stored (the next
            #                buffer slot belongs to a *new* episode), so instead of
            #                forcing 0 we bootstrap with the adjacent-state value.
            #   otherwise -> V(s_{t+1}) (the next transition's value)
            if is_terminal:
                boot = 0.0
            elif is_truncated:
                boot = values[t]
            else:
                boot = next_value

            delta = rewards[t] + self.cfg.gamma * boot - values[t]

            # GAE resets at every episode boundary (terminal *or* truncated).
            is_boundary = float(is_terminal or is_truncated)
            gae = (
                delta + self.cfg.gamma * self.cfg.gae_lambda * (1.0 - is_boundary) * gae
            )
            advantages[t] = gae
            next_value = values[t]

        return advantages, advantages + values
