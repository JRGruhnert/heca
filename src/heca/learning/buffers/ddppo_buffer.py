from dataclasses import dataclass
from pathlib import Path
from collections import defaultdict
import torch
from heca.learning.buffers.buffer import Buffer, BufferData
from heca.misc.hardware import device
from heca.misc import logger
from torch_geometric.data import HeteroData


class DDPPOBuffer(Buffer):
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
        caps = self.bucket_capacity
        return all(len(self.buckets[tag]) >= caps[tag] for tag in caps)

    def __len__(self) -> int:
        return sum(len(b) for b in self.buckets.values())

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

    def store_feedback(self, reward, terminal, truncated, tag):
        self.buckets[tag][-1].reward = reward
        self.buckets[tag][-1].terminal = terminal
        self.buckets[tag][-1].truncated = truncated
        self.trim_if_overflow(tag)

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

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.trim_to_exact_capacity()
        all_adv, all_ret = [], []
        for tag in self.sorted_tags:
            bucket = self.buckets[tag]
            rewards = [d.reward for d in bucket]
            terminals = [d.terminal for d in bucket]
            values = [d.value.item() for d in bucket]
            advs, rets = self._gae_for_bucket(rewards, terminals, values)
            all_adv.extend(advs)
            all_ret.extend(rets)
        return (
            torch.tensor(all_adv, dtype=torch.float32, device=device),
            torch.tensor(all_ret, dtype=torch.float32, device=device),
        )

    def _gae_for_bucket(self, rewards, terminals, values):
        T = len(rewards)
        advantages = [0.0] * T
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
        returns = [adv + values[t] for t, adv in enumerate(advantages)]
        return advantages, returns

    def flush_and_rate(self) -> bool:
        total_terminals = 0
        total_dones = 0
        for bucket in self.buckets.values():
            for d in bucket:
                if d.truncated:
                    total_terminals += 1
                    if d.terminal:
                        total_dones += 1
        if total_terminals == 0:
            return False
        current = total_dones / total_terminals
        self.buckets.clear()
        if current > self.highscore:
            self.highscore = current
            return True
        return False

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
