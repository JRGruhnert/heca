from dataclasses import dataclass
from typing import cast
import torch

from heca.learning.server import FLServer
from heca.learning.ppo import PPO


class FPPO(PPO):
    @dataclass(kw_only=True)
    class Config(PPO.Config):
        server: FLServer.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.server = FLServer.get(cfg.server)
        self.server.register(self.cfg.tag)
        self._last_version = self.server.version
        self._sync_keys = self.network.sync_keys(self.server.cfg.network.sync)
        self._load_global(
            self.server.sync_state_dict(self.server.global_network.state_dict())
        )
        self._global_params = self._global_snapshot()

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

    def _fedprox_term(self) -> torch.Tensor:
        loss = 0.0
        for name, local_p in self.network.named_parameters():
            global_p = self._global_params.get(name)
            if global_p is None:
                continue
            loss += torch.sum((local_p - global_p) ** 2)  # euklidische distanz squared
        return (self.server.cfg.fedprox_mu / 2) * loss  # type: ignore

    def sync(self):
        state_dict = self.server.submit(
            self.cfg.tag,
            cast(dict[str, torch.Tensor], self.network.state_dict()),
            self._last_version,
        )
        self._load_global(state_dict)
        self._global_params = self._global_snapshot()
        self.inference_net.load_state_dict(self.network.state_dict())
        self._last_version = self.server.version
        for pg in self.optim.param_groups:
            for p in pg.get("params", []):
                s = self.optim.state.get(p)
                if s is not None and "exp_avg" in s:
                    s["exp_avg"].zero_()
                    s["exp_avg_sq"].zero_()
