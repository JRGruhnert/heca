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
        self.network.load_state_dict(self.server.global_network.state_dict())
        # Detached snapshot of the global weights used by the fedprox term. It
        # is only swapped in ``sync`` (on the event loop) and read by
        # ``_fedprox_term`` (in a worker thread), so it never races with the
        # server's live ``global_network``.
        self._global_params = [p.detach().clone() for p in self.network.parameters()]

    def _fedprox_term(self) -> torch.Tensor:
        loss = 0.0
        for local_p, global_p in zip(self.network.parameters(), self._global_params):
            loss += torch.sum((local_p - global_p) ** 2)  # euklidische distanz squared
        return (self.server.cfg.fedprox_mu / 2) * loss  # type: ignore

    async def sync(self):
        state_dict = await self.server.submit(
            self.cfg.tag,
            cast(dict[str, torch.Tensor], self.network.state_dict()),
            self._last_version,
        )
        self.network.load_state_dict(state_dict)
        self._global_params = [p.detach().clone() for p in self.network.parameters()]
        self.inference_net.load_state_dict(self.network.state_dict())
        self._last_version = self.server.version
        for pg in self.optim.param_groups:
            for p in pg.get("params", []):
                s = self.optim.state.get(p)
                if s is not None and "exp_avg" in s:
                    s["exp_avg"].zero_()
                    s["exp_avg_sq"].zero_()
