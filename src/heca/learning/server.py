import asyncio
from dataclasses import dataclass
from collections import OrderedDict
from pathlib import Path
import torch

from heca.heca_gnn.network import Network
from heca.misc import hardware, logger
from heca.misc.base import Persistable


class FLServer(Persistable):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        folder: str = "network"
        label: str = "federated"
        network: Network.Config
        fedavgm_beta: float = 0.9
        max_update: int = 1000
        fedprox_mu: float = 0.01
        save_interval: int = 50

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.global_network = Network.get(cfg.network)
        for p in self.global_network.parameters():
            p.requires_grad = False
        self._momentum: dict[str, torch.Tensor] = {}
        self._momentum_beta = cfg.fedavgm_beta
        self._version = 0
        self._clients: set[str] = set()
        self._pending: dict[str, dict[str, torch.Tensor]] = {}
        self._cond = asyncio.Condition()

    def aggregate(self, state_dicts: list[dict[str, torch.Tensor]]):
        avg = self.fedavg(state_dicts)
        self.global_network.load_state_dict(avg)

    def aggregate_with_momentum(self, state_dicts: list[dict[str, torch.Tensor]]):
        """FedAvgM — server momentum on weight deltas.

        ``new = prev + m`` where ``m = β·m + (1-β)·Δ`` and ``Δ`` is the
        FedAvg step from the current global model.
        """
        avg = self.fedavg(state_dicts)
        current = self.global_network.state_dict()

        delta = {}
        for k in avg:
            cur = current[k].to(avg[k].dtype, device=avg[k].device)
            delta[k] = avg[k] - cur

        if not self._momentum:
            self._momentum = delta
        else:
            for k in delta:
                self._momentum[k] = (
                    self._momentum_beta * self._momentum[k]
                    + (1.0 - self._momentum_beta) * delta[k]
                )

        new_weights = OrderedDict()
        for k in avg:
            cur = current[k].to(avg[k].dtype, device=avg[k].device)
            new_weights[k] = cur + self._momentum[k]
        self.global_network.load_state_dict(new_weights)

    def aggregate_weighted(
        self,
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ):
        avg = self.fedavg_weighted(state_dicts, weights)
        self.global_network.load_state_dict(avg)

    def register(self, tag: str):
        """Register a client before training starts. Must be called for every
        client before the first ``submit``."""
        self._clients.add(tag)

    async def submit(
        self,
        tag: str,
        state_dict: dict[str, torch.Tensor],
        last_version: int,
    ) -> dict[str, torch.Tensor]:
        """Deposit weights and block until the round's aggregation is ready.

        The last registered client to submit triggers the aggregation and
        wakes everyone. Returns the fresh global state_dict.
        """
        assert tag in self._clients, f"Unregistered client {tag}"
        async with self._cond:
            self._pending[tag] = state_dict
            if len(self._pending) >= len(self._clients):
                self._aggregate_round()
                self._version += 1
                if self._version % self.cfg.save_interval == 0:
                    self.save()
                self._cond.notify_all()
            else:
                while self._version <= last_version:
                    await self._cond.wait()
            return self.global_network.state_dict()

    def _aggregate_round(self):
        state_dicts = [self._pending[k] for k in sorted(self._pending)]
        self.aggregate_with_momentum(state_dicts)
        self._pending.clear()

    @property
    def version(self) -> int:
        return self._version

    def _save(self, path: Path):
        filepath = path / f"checkpoint_{self._version}.pt"

        checkpoint = {
            "global_network": self.global_network.state_dict(),
            "momentum": self._momentum,
            "version": self._version,
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Saved global checkpoint to {filepath}")

    def _load(self, path: Path):
        filepath = path / "checkpoint.pt"
        if not filepath.exists():
            logger.warning(f"No checkpoint found at {filepath}. Starting from scratch.")
            return

        checkpoint = torch.load(filepath, map_location=hardware.device)

        self.global_network.load_state_dict(checkpoint["global_network"])
        self._momentum = checkpoint.get("momentum", {})
        self._version = checkpoint.get("version", 0)
        logger.info(f"Loaded global checkpoint from {filepath} (v{self._version})")

    @staticmethod
    def fedavg(
        state_dicts: list[dict[str, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        if not state_dicts:
            raise ValueError("state_dicts must not be empty")

        avg = {}
        for key in state_dicts[0]:
            stacked = torch.stack([sd[key].float() for sd in state_dicts])
            avg[key] = stacked.mean(dim=0)

        first = state_dicts[0]
        result = OrderedDict()
        for key in first:
            result[key] = avg[key].to(dtype=first[key].dtype, device=first[key].device)

        return result

    @staticmethod
    def fedavg_weighted(
        state_dicts: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if len(state_dicts) != len(weights):
            raise ValueError("len(state_dicts) must equal len(weights)")
        if not state_dicts:
            raise ValueError("state_dicts must not be empty")

        total_w = sum(weights)
        normalized = [w / total_w for w in weights]

        avg = {}
        for key in state_dicts[0]:
            weighted_sum = sum(
                sd[key].float() * w for sd, w in zip(state_dicts, normalized)
            )
            avg[key] = weighted_sum

        first = state_dicts[0]
        result = OrderedDict()
        for key in first:
            result[key] = avg[key].to(dtype=first[key].dtype, device=first[key].device)

        return result
