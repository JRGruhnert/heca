from dataclasses import dataclass
from pathlib import Path
import torch

from heca.learning.buffers.buffer import Buffer, BufferData
from heca.misc.hardware import device
from heca.misc import logger
from torch_geometric.data import HeteroData


class PPOBuffer(Buffer):
    @dataclass(kw_only=True)
    class Config(Buffer.Config):
        capacity: int = 2048

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.bucket: list[BufferData] = []
        self.highscore = float("-inf")

    @property
    def full(self) -> bool:
        return len(self.bucket) >= self.cfg.capacity

    @property
    def flat_data(self) -> list[HeteroData]:
        return [d.data for d in self.bucket]

    @property
    def flat_actions(self) -> torch.Tensor:
        return [d.action for d in self.bucket]

    @property
    def flat_logprobs(self) -> torch.Tensor:
        return [d.logprobs for d in self.bucket]

    @property
    def flat_values(self) -> torch.Tensor:
        return [d.values for d in self.bucket]

    def store_prediction(self, data, action, logprob, value, tag):
        self.bucket.append(BufferData(data, action, logprob, value))

    def store_feedback(self, reward, terminal, truncated, tag):
        self.bucket[-1].reward = reward
        self.bucket[-1].terminal = terminal
        self.bucket[-1].truncated = truncated

    def trim_to_exact_capacity(self):
        self.bucket = self.bucket[-self.cfg.capacity :]

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.trim_to_exact_capacity()
        rewards = [d.reward for d in self.bucket]
        terminals = [d.terminal for d in self.bucket]
        values = [d.value.item() for d in self.bucket]
        advs, rets = self._gae(rewards, terminals, values)
        return (
            torch.tensor(advs, dtype=torch.float32, device=device),
            torch.tensor(rets, dtype=torch.float32, device=device),
        )

    def _gae(self, rewards, terminals, values):
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
        total_truncates = 0
        total_terminals = 0
        for d in self.bucket:
            if d.truncated:
                total_truncates += 1
                if d.terminal:
                    total_terminals += 1
        if total_truncates == 0:
            return False
        current = total_terminals / total_truncates
        self.bucket.clear()
        if current > self.highscore:
            self.highscore = current
            return True
        return False

    def save(self, path: Path, label: str):
        logger.info(f"Saving buffer '{label}'")
        serialized = {
            "data": [d.data for d in self.bucket],
            "actions": torch.stack([d.action for d in self.bucket]),
            "logprobs": torch.stack([d.logprob for d in self.bucket]),
            "values": torch.stack([d.value for d in self.bucket]),
            "rewards": torch.tensor([d.reward for d in self.bucket]),
            "dones": torch.tensor([d.terminal for d in self.bucket]),
            "terminals": torch.tensor([d.truncated for d in self.bucket]),
        }
        torch.save(serialized, path / f"buffer_{label}.pt")

    def load(self, path: Path, label: str):
        logger.info(f"Loading buffer '{label}'")
        serialized = torch.load(path / f"buffer_{label}.pt")
        self.bucket.clear()
        for tag, data in serialized.items():
            bucket = []
            actions = torch.unbind(data["actions"])
            logprobs = torch.unbind(data["logprobs"])
            values = torch.unbind(data["values"])
            rewards = data["rewards"].tolist()
            dones = data["dones"].tolist()
            terminals = data["terminals"].tolist()
            for i in range(len(rewards)):
                self.bucket.append(
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
            self.bucket[tag] = bucket
