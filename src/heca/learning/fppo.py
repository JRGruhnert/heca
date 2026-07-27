from dataclasses import dataclass
from collections import OrderedDict
import torch

from heca.heca_gnn.network import Network
from heca.heca_gnn.network1 import Network1
from heca.learning.ppo import PPO
from heca.misc.base import Registerable


class FPPO(PPO):
    @dataclass(kw_only=True)
    class Config(PPO.Config):
        label: str = "fppo"
        fedprox_mu: float = 0.01
        server: Server.Config = Server.Config(label="global")

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.server = Server.get(cfg.server)

    def _fedprox_term(self) -> torch.Tensor:
        if self.cfg.fedprox_mu == 0:
            return torch.tensor(0.0, device=next(self.network.parameters()).device)

        loss = 0.0
        for local_p, global_p in zip(
            self.network.parameters(), self.server.global_network.parameters()
        ):
            loss += torch.sum((local_p - global_p) ** 2)
        return (self.cfg.fedprox_mu / 2) * loss

    def pull_global(self):
        self.network.load_state_dict(self.server.global_network.state_dict())
        for pg in self.optim.param_groups:
            for p in pg.get("params", []):
                s = self.optim.state.get(p)
                if s is not None and "exp_avg" in s:
                    s["exp_avg"].zero_()
                    s["exp_avg_sq"].zero_()
