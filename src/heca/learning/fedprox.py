from dataclasses import dataclass

import torch

from heca.learning.fppo import FPPO


class FedProxPPO(FPPO):
    @dataclass(kw_only=True)
    class Config(FPPO.Config):
        mu: float = 0.01

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    def learn(self):
        super().learn()
        # measured once, after the whole local phase: the penalty logged inside the
        # minibatch loop is a mean over parameter states, from no drift (first batch
        # of an epoch) to the largest drift (last one), so it understates this
        self.metrics["fedprox/drift_norm"] = self.drift_norm()

    def _fedprox_term(self) -> torch.Tensor:
        return self._proximal_term(self.network, self.cfg.mu)
