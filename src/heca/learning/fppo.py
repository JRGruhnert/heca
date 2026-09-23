from dataclasses import dataclass

import torch

from heca.learning import dist as pdist
from heca.learning.ppo import PPO
from heca.misc import hardware


class FPPO(PPO):
    @dataclass(kw_only=True)
    class Config(PPO.Config):
        # server-side FedAvgM momentum
        fedavgm_beta: float = 0.9

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self._sync_keys = self.network.sync_keys(cfg.network.sync)
        self._beta = cfg.fedavgm_beta
        self._momentum: dict[str, torch.Tensor] = {}
        self._version = 0
        # rank 0's weights are the initial global model for every rank
        self._global = self._synced_state()
        pdist.broadcast_tensors(self._global)
        self._load_global(self._global)
        self._global_params = self._global_snapshot()

    def _synced_state(self) -> dict[str, torch.Tensor]:
        return {
            name: t.detach().to(device="cpu", dtype=torch.float32, copy=True)
            for name, t in self.network.state_dict().items()
            if name in self._sync_keys
        }

    def _global_snapshot(self) -> dict[str, torch.Tensor]:
        return {
            name: p.detach().clone()
            for name, p in self.network.named_parameters()
            if name in self._sync_keys
        }

    def _load_global(self, state_dict: dict[str, torch.Tensor]) -> None:
        self.network.load_state_dict(
            {k: v for k, v in state_dict.items() if k in self._sync_keys}, strict=False
        )

    def _proximal_term(
        self,
        net: torch.nn.Module,
        coef: float,
        anchor: dict[str, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if coef == 0.0:
            return torch.tensor(0.0, device=hardware.device)
        anchor = self._global_params if anchor is None else anchor
        penalty = torch.zeros((), device=hardware.device)
        for name, param in net.named_parameters():
            shared = anchor.get(name)
            if shared is None:
                continue
            penalty = penalty + torch.sum((param - shared) ** 2)
        return (coef / 2) * penalty

    def sync(self):
        # a lone client is its own aggregate, so there is nothing to reduce
        if pdist.world_size() > 1:
            self._aggregate_distributed()
        self._global_params = self._global_snapshot()
        self._sync_inference()
        for pg in self.optim.param_groups:
            for p in pg.get("params", []):
                s = self.optim.state.get(p)
                if s is not None and "exp_avg" in s:
                    s["exp_avg"].zero_()
                    s["exp_avg_sq"].zero_()
        self._periodic_save()

    def _aggregate_distributed(self) -> None:
        """FedAvgM over ranks: averaged weights plus momentum on the update."""
        avg = pdist.mean_tensors(self._synced_state())
        delta = {k: avg[k] - self._global[k].to(dtype=avg[k].dtype) for k in avg}
        if not self._momentum:
            self._momentum = {k: v.clone() for k, v in delta.items()}
        else:
            for k, v in delta.items():
                self._momentum[k] = (
                    self._beta * self._momentum[k] + (1.0 - self._beta) * v
                )
        self._global = {
            k: self._global[k].to(dtype=avg[k].dtype) + self._momentum[k] for k in avg
        }
        self._version += 1
        self._load_global(self._global)
