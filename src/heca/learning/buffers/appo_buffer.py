from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import torch
from heca.learning.buffers.buffer import Buffer, BufferData
from heca.misc.hardware import device
from heca.misc import logger
from torch_geometric.data import HeteroData


class APPOBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 2048

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.buckets: dict[str, list[BufferData]] = defaultdict(list)
        self.highscore = float("-inf")

    @property
    def sorted_tags(self) -> list[str]:
        return sorted(self.buckets.keys())

    @property
    def base(self) -> int:
        n = len(self.buckets)
        return self.cfg.capacity // n if n else 0

    @property
    def rem(self) -> int:
        n = len(self.buckets)
        return self.cfg.capacity % n if n else 0

    @property
    def bucket_capacity(self) -> dict[str, int]:
        bc = {}
        for i, tag in enumerate(self.sorted_tags):
            bc[tag] = self.base + (1 if i < self.rem else 0)
        return bc

    @property
    def full(self) -> bool:
        return sum(len(b) for b in self.buckets.values()) >= self.cfg.capacity

    @property
    def flat_data(self) -> list[HeteroData]:
        flat = []
        for tag in self.sorted_tags:
            flat.extend(d.data for d in self.buckets[tag])
        return flat

    @property
    def flat_actions(self) -> torch.Tensor:
        tensors = []
        for tag in self.sorted_tags:
            tensors.extend(d.action for d in self.buckets[tag])
        return torch.stack(tensors, dim=0).to(device)

    @property
    def flat_logprobs(self) -> torch.Tensor:
        tensors = []
        for tag in self.sorted_tags:
            tensors.extend(d.logprob for d in self.buckets[tag])
        return torch.stack(tensors, dim=0).to(device)

    @property
    def flat_values(self) -> torch.Tensor:
        tensors = []
        for tag in self.sorted_tags:
            tensors.extend(d.value for d in self.buckets[tag])
        return torch.stack(tensors, dim=0).to(device)

    def store_prediction(self, data, action, logprob, value, tag):
        self.buckets[tag].append(BufferData(data, action, logprob, value))

    def enforce_fifo_capacity(self):
        raise NotImplementedError

    def store_feedback(self, reward, terminal, truncated, tag) -> bool:
        self.buckets[tag][-1].reward = reward
        self.buckets[tag][-1].terminal = terminal
        self.buckets[tag][-1].truncated = truncated

        # Enforce FIFO capacity
        self.enforce_fifo_capacity()
        return self.full

    def _find_first_episode_end(self, bucket: list[BufferData], start=0) -> int:
        for i in range(start, len(bucket)):
            if bucket[i].terminal or bucket[i].truncated:
                return i
        return -1

    def trim_if_overflow(self, tag: str):
        bucket = self.buckets.get(tag, [])
        target = self.bucket_capacity.get(tag, self.cfg.capacity)
        while len(bucket) > target:
            end_idx = self._find_first_episode_end(bucket)
            if end_idx == -1:
                break
            ep_len = end_idx + 1
            overflow = len(bucket) - target
            if ep_len <= overflow:
                bucket = bucket[ep_len:]
                self.buckets[tag] = bucket
            else:
                break

    def trim_to_exact_capacity(self):
        for tag, bucket in self.buckets.items():
            target = self.bucket_capacity.get(tag, self.cfg.capacity)
            if len(bucket) > target:
                # Keep the newest 'target' steps (most recent)
                self.buckets[tag] = bucket[-target:]

    def compute_advantages(self, current_logprobs, current_values):
        """V-trace for APPO. Requires current policy outputs."""
        self.enforce_fifo_capacity()
        all_adv, all_ret = [], []
        offset = 0
        for tag in self.sorted_tags:
            bucket = self.buckets[tag]
            T = len(bucket)
            if T == 0:
                continue
            cp = current_logprobs[offset : offset + T]
            cv = current_values[offset : offset + T]
            offset += T
            rewards = [d.reward for d in bucket]
            terminals = [d.terminal or d.truncated for d in bucket]
            behavior_lp = (
                torch.stack([d.logprob for d in bucket]).squeeze(-1).to(device)
            )
            behavior_v = torch.stack([d.value for d in bucket]).squeeze(-1).to(device)
            advs, rets = self._vtrace(
                rewards, terminals, behavior_lp, behavior_v, cp, cv
            )
            all_adv.append(advs)
            all_ret.append(rets)
        return torch.cat(all_adv), torch.cat(all_ret)

    def _vtrace(
        self,
        rewards,
        terminals,
        behavior_logprobs,
        behavior_values,
        current_logprobs,
        current_values,
    ):
        """V-trace for one contiguous episode.

        All tensors are shape [T].
        rewards/terminals are Python lists.
        """
        T = len(rewards)
        device = current_logprobs.device

        ratios = torch.exp(current_logprobs - behavior_logprobs)  # π / μ
        rho_bar = 1.0
        c_bar = 1.0

        advantages = torch.zeros(T, device=device)
        returns = torch.zeros(T, device=device)

        # ── Bootstrap V(s_T) ──────────────────────────────
        # V(s_T) = 0 if the last step ended in terminal,
        #         else bootstrap with V_current(s_{T-1}) (approx)
        if terminals[-1]:
            next_val = 0.0
        else:
            next_val = current_values[-1].item()
        v_next = torch.tensor(next_val, device=device)  # v_T

        for t in reversed(range(T)):
            rho = torch.clamp(ratios[t], max=rho_bar)  # ρ_t
            c = torch.clamp(ratios[t], max=c_bar)  # c_t
            term = float(terminals[t])  # 1 if s_{t+1} is terminal

            # TD error: δ_t = ρ_t · (r_t + γ·V(s_{t+1}) - V(s_t))
            delta = rho * (
                rewards[t] + self.cfg.gamma * next_val * (1 - term) - current_values[t]
            )

            # V-trace target: v_t = V(s_t) + δ_t + γ·c_t·(1-term)·(v_{t+1} - V(s_{t+1}))
            if t < T - 1:
                v_t = (
                    current_values[t]
                    + delta
                    + self.cfg.gamma * c * (1 - term) * (v_next - current_values[t + 1])
                )
            else:
                v_t = current_values[t] + delta  # last step, no correction from future

            returns[t] = v_t
            advantages[t] = v_t - current_values[t]

            # ── Prepare for next (previous) step ─────────
            # For step t-1, V(s_t) is the bootstrap value.
            # s_t is terminal iff terminals[t-1] is True.
            if t > 0:
                next_val = 0.0 if terminals[t - 1] else current_values[t].item()
            v_next = v_t

        return advantages, returns

    def save(self, path: Path, label: str):
        logger.info(f"Saving buffer '{label}'")
        serialized = {}
        for tag, bucket in self.buckets.items():
            serialized[tag] = {
                "data": [d.data for d in bucket],
                "actions": torch.stack([d.action for d in bucket]),
                "logprobs": torch.stack([d.logprob for d in bucket]),
                "values": torch.stack([d.value for d in bucket]),
                "rewards": torch.tensor([d.reward for d in bucket]),
                "dones": torch.tensor([d.terminal for d in bucket]),
                "terminals": torch.tensor([d.truncated for d in bucket]),
            }
        torch.save(serialized, path / f"buffer_{label}.pt")

    def load(self, path: Path, label: str):
        logger.info(f"Loading buffer '{label}'")
        serialized = torch.load(path / f"buffer_{label}.pt")
        self.buckets.clear()
        for tag, data in serialized.items():
            bucket = []
            actions = torch.unbind(data["actions"])
            logprobs = torch.unbind(data["logprobs"])
            values = torch.unbind(data["values"])
            rewards = data["rewards"].tolist()
            dones = data["dones"].tolist()
            terminals = data["terminals"].tolist()
            for i in range(len(rewards)):
                bucket.append(
                    BufferData(
                        data=data["data"][i],
                        action=actions[i],
                        logprob=logprobs[i],
                        value=values[i],
                        reward=rewards[i],
                        terminal=dones[i],
                        truncated=terminals[i],
                    )
                )
            self.buckets[tag] = bucket
