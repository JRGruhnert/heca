from dataclasses import dataclass
import torch

from heca.learning.buffers.buffer import Buffer, BufferData


class StreamBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 256
        gamma: float = 0.99

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def is_allowed(self, data: BufferData) -> bool:
        return True  # Stream Buffer does not reject

    def compute_advantages(
        self, current_logprobs, current_values
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """V-trace for the entire buffer."""
        """
        Compute V-trace for a list of BufferData items.

        Args:
            items: List[BufferData] in **chronological order** (global FIFO order).
            current_logprobs: Tensor of shape [len(items), ...] aligned with `items`.
            current_values: Tensor of shape [len(items)] aligned with `items`.

        Returns:
            advantages, returns: Tensors of shape [len(items)].
        """
        # Compute V-trace independently for each contiguous tag sequence
        rewards = [d.reward for d in self.queue]
        terminals = [d.terminal or d.truncated for d in self.queue]
        behavior_lp = torch.stack([d.logprob for d in self.queue]).detach().squeeze(-1)
        behavior_v = torch.stack([d.value for d in self.queue]).detach().squeeze(-1)

        return self._vtrace(
            rewards,
            terminals,
            behavior_lp,
            behavior_v,
            current_logprobs,
            current_values,
        )

    def _vtrace(
        self,
        rewards,
        terminals,
        behavior_logprobs,
        behavior_values,
        current_logprobs,
        current_values,
    ):
        """V-trace for one contiguous tag sequence.

        All tensors are shape [T].
        rewards/terminals are Python lists.
        """
        T = len(rewards)
        ratios = torch.exp(current_logprobs - behavior_logprobs)  # π / μ
        advantages = torch.zeros(T)
        returns = torch.zeros(T)
        next_val = 0.0 if terminals[-1] else current_values[-1].item()
        v_next = torch.tensor(next_val)

        for i in reversed(range(T)):
            rho = torch.clamp(ratios[i], max=1.0)
            c = torch.clamp(ratios[i], max=1.0)
            term = float(terminals[i])
            # TD error
            delta = rho * (
                rewards[i] + self.cfg.gamma * next_val * (1 - term) - current_values[i]
            )
            # V-trace target
            if i < T - 1:
                v_t = (
                    current_values[i]
                    + delta
                    + self.cfg.gamma * c * (1 - term) * (v_next - current_values[i + 1])
                )
            else:
                v_t = current_values[i] + delta
            returns[i] = v_t
            advantages[i] = v_t - current_values[i]
            if i > 0:
                next_val = 0.0 if terminals[i - 1] else current_values[i].item()
            v_next = v_t

        return advantages, returns
