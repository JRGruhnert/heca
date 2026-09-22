import copy
from dataclasses import dataclass

import torch

from heca.heca_gnn.network import Network
from heca.learning.fedprox import FedProxPPO


class DittoPPO(FedProxPPO):
    @dataclass(kw_only=True)
    class Config(FedProxPPO.Config):
        mu: float = 0.0
        personal_coef: float = 0.1
        personal_lr: float | None = None
        act_personal: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        # independent copy: never uploaded, never overwritten by the server
        self.personal: Network = copy.deepcopy(self.network)
        self.personal_optim = torch.optim.AdamW(
            self.personal.parameters(),
            lr=cfg.personal_lr if cfg.personal_lr is not None else cfg.lr,
        )

    def _personal_term(self) -> torch.Tensor:
        """``lambda/2 * ||v_k - w||^2`` over the federated tensors.

        ``_global_params`` is the shared model as of the last aggregation, so
        during one local phase the personal model is pulled towards a fixed
        anchor. Only tensors the server aggregates are penalized; the local
        heads are personal either way.
        """
        return self._proximal_term(self.personal, self.cfg.personal_coef)

    def learn(self):
        """One local phase: the federated model first, then the personal one."""
        adv, rtn = self.buffer.compute_advantages()

        self._mini_batch_loop(adv, rtn)
        self._anneal()

        if self.cfg.personal_coef > 0.0:
            self._mini_batch_loop(
                adv,
                rtn,
                net=self.personal,
                optim=self.personal_optim,
                penalty=self._personal_term,
                prefix="personal/",
                penalty_key="ditto_penalty",
            )
            self._anneal(self.personal_optim)

    def _sync_inference(self):
        """Let the personal model collect experience, if asked to."""
        source = self.personal if self.cfg.act_personal else self.network
        self.inference_net.load_state_dict(source.state_dict())

    def _checkpoint(self) -> dict:
        return {
            **super()._checkpoint(),
            "personal_network": self.personal.state_dict(),
            "personal_optimizer": self.personal_optim.state_dict(),
        }

    def _restore(self, checkpoint: dict):
        super()._restore(checkpoint)
        if "personal_network" in checkpoint:
            self.personal.load_state_dict(checkpoint["personal_network"])
            self.personal_optim.load_state_dict(checkpoint["personal_optimizer"])
        # start the next local phase anchored at the restored shared model
        self._global_params = self._global_snapshot()
        self._sync_inference()
