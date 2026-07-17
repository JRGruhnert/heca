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
        adv_flat = torch.zeros(len(self.queue))
        ret_flat = torch.zeros(len(self.queue))

        # Compute V-trace independently for each contiguous tag sequence
        for _, indices in self.tag_indices.items():
            group_items = [self.queue[i] for i in indices]
            rewards = [d.reward for d in group_items]
            terminals = [d.terminal or d.truncated for d in group_items]
            behavior_lp = (
                torch.stack([d.logprob for d in group_items]).detach().squeeze(-1)
            )
            behavior_v = (
                torch.stack([d.value for d in group_items]).detach().squeeze(-1)
            )

            cp = current_logprobs[indices]
            cv = current_values[indices]

            advs, rets = self._vtrace(
                rewards, terminals, behavior_lp, behavior_v, cp, cv
            )

            adv_flat[indices] = advs
            ret_flat[indices] = rets

        return adv_flat, ret_flat

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
