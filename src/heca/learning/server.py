from dataclasses import dataclass
from collections import OrderedDict
import torch

from heca.heca_gnn.network import Network
from heca.heca_gnn.network2 import Network2
from heca.misc.base import Registerable


class FLServer(Registerable):
    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        network: Network.Config = Network2.Config()
        fedavgm_beta: float = 0.9
        max_update: int = 1000
        fedprox_mu: float = 0.01

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.global_network = Network.get(cfg.network)
        for p in self.global_network.parameters():
            p.requires_grad = False
        self._momentum: dict[str, torch.Tensor] = {}
        self._momentum_beta = cfg.fedavgm_beta

    def aggregate(self, state_dicts: list[OrderedDict[str, torch.Tensor]]):
        avg = self.fedavg(state_dicts)
        self.global_network.load_state_dict(avg)

    def aggregate_with_momentum(
        self, state_dicts: list[OrderedDict[str, torch.Tensor]]
    ):
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
        state_dicts: list[OrderedDict[str, torch.Tensor]],
        weights: list[float],
    ):
        avg = self.fedavg_weighted(state_dicts, weights)
        self.global_network.load_state_dict(avg)

    @staticmethod
    def fedavg(
        state_dicts: list[OrderedDict[str, torch.Tensor]],
    ) -> OrderedDict[str, torch.Tensor]:
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
        state_dicts: list[OrderedDict[str, torch.Tensor]],
        weights: list[float],
    ) -> OrderedDict[str, torch.Tensor]:
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

    def has_update(self) -> bool:
        raise NotImplementedError
