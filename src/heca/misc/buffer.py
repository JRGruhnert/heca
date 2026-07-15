from pathlib import Path
import torch

from heca.agents.agent import AgentFeedback
from heca.misc.hardware import device
from heca.misc import logger
from torch_geometric.data import HeteroData


class RolloutBuffer:
    """Stores (s, a, logprob, value, reward, terminal) for one PPO policy.

    Multiple agents can feed into the same buffer. GAE computation
    respects episode boundaries across agents.
    """

    def __init__(self, capacity: int, gamma: float, gae_lambda: float):
        self.capacity = capacity
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # Per-step storage
        self.data: list[HeteroData] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.terminals: list[bool] = []
        self.agent_ids: list[str] = []  # optional per-agent tracking

        # Statistics
        self.highscore = float("-inf")

    # --- Size ---

    def __len__(self) -> int:
        return len(self.data)

    @property
    def full(self) -> bool:
        return len(self) >= self.capacity

    # --- Store ---

    def store_action(
        self,
        data: HeteroData,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        agent_id: str = "",
    ):
        self.data.append(data)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.agent_ids.append(agent_id)

    def store_feedback(self, fb: AgentFeedback, agent_id: str = ""):
        self.rewards.append(fb.reward)
        self.dones.append(fb.done)
        self.terminals.append(fb.terminal)
        # agent_id for the reward step — assumes store_action was called first
        # so the agent_id list already has an entry for this step
        if agent_id:
            self.agent_ids[-1] = agent_id

    def clear(self):
        self.data.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.terminals.clear()
        self.agent_ids.clear()

    def compute_gae(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Group by agent_id, compute GAE per group, remap to flat buffer order."""
        from collections import defaultdict

        # 1. Group indices by agent_id
        groups: dict[str, list[int]] = defaultdict(list)
        for i, aid in enumerate(self.agent_ids):
            groups[aid].append(i)

        # 2. Compute GAE per group, store result per index
        adv_per_index: dict[int, float] = {}
        rtn_per_index: dict[int, float] = {}
        for aid, indices in groups.items():
            advs, rtns = self._gae_for_indices(indices)
            for idx, adv, rtn in zip(indices, advs, rtns):
                adv_per_index[idx] = adv
                rtn_per_index[idx] = rtn

        # 3. Remap to flat order (preserving buffer chronology)
        B = len(self)
        adv_flat = [adv_per_index[i] for i in range(B)]
        rtn_flat = [rtn_per_index[i] for i in range(B)]

        return (
            torch.tensor(adv_flat, dtype=torch.float32, device=device),
            torch.tensor(rtn_flat, dtype=torch.float32, device=device),
        )

    def _gae_for_indices(self, indices: list[int]) -> tuple[list[float], list[float]]:
        B = len(indices)
        # Convert to float scalars immediately
        padded_vals = [self.values[i].item() for i in indices] + [0.0]

        advantages: list[float] = []
        gae = 0.0
        for t in reversed(range(B)):
            idx = indices[t]
            terminal = float(self.terminals[idx])
            delta = (
                self.rewards[idx]
                + self.gamma * padded_vals[t + 1] * (1 - terminal)
                - padded_vals[t]
            )
            gae = delta + self.gamma * self.gae_lambda * (1 - terminal) * gae
            advantages.insert(0, gae)

        returns = [
            adv + self.values[idx].item() for adv, idx in zip(advantages, indices)
        ]
        return advantages, returns

    def flush_and_rate(self) -> bool:
        """Clear the buffer and return True if success rate is a new high."""
        if not self.terminals:
            return False
        current = sum(self.dones) / len(self.terminals)
        self.clear()
        if current > self.highscore:
            self.highscore = current
            return True
        return False

    def save(self, path: Path, tag: str):
        logger.info(f"Saving buffer '{tag}' ({len(self)} steps)")
        torch.save(
            {
                "actions": torch.stack(self.actions),
                "logprobs": torch.stack(self.logprobs),
                "values": torch.stack(self.values),
                "rewards": torch.tensor(self.rewards),
                "dones": torch.tensor(self.dones),
                "terminals": torch.tensor(self.terminals),
                "agent_ids": self.agent_ids,
            },
            path / tag / f"buffer.pt",
        )

    def load(self, path: Path, tag: str):
        logger.info(f"Loading buffer '{tag}'")
        data = torch.load(path / tag / f"buffer.pt")
        self.data.clear()
        self.actions = list(torch.unbind(data["actions"]))
        self.logprobs = list(torch.unbind(data["logprobs"]))
        self.values = list(torch.unbind(data["values"]))
        self.rewards = data["rewards"].tolist()
        self.dones = data["dones"].tolist()
        self.terminals = data["terminals"].tolist()
        self.agent_ids = list(data["agent_ids"])
