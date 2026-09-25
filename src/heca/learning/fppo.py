from dataclasses import dataclass
from hashlib import sha1
from pathlib import Path

import torch

from heca.learning import dist as pdist
from heca.learning.ppo import PPO
from heca.misc import hardware, logger


class FPPO(PPO):
    @dataclass(kw_only=True)
    class Config(PPO.Config):
        fedavgm_beta: float = 0.9
        server_lr: float | None = None
        k: int = 1

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self._sync_keys = self.network.sync_keys(cfg.network.sync)
        self._beta = cfg.fedavgm_beta
        if not 0.0 <= self._beta < 1.0:
            raise ValueError(f"fedavgm_beta must be in [0, 1), got {self._beta}")
        if cfg.server_lr is not None and cfg.server_lr < 0.0:
            raise ValueError(f"server_lr must be >= 0 or None, got {cfg.server_lr}")
        self._momentum: dict[str, torch.Tensor] = {}
        self._version = 0
        self._rounds = 0
        self._aggregated = False
        self._shared_saved_at = 0
        # rank 0's weights are the initial global model for every rank
        self._global = self._synced_state()
        pdist.broadcast_tensors(self._global)
        self._load_global(self._global)
        self._global_params = self._global_snapshot()
        logger.info(
            f"[{cfg.tag}] server: FedAvgM beta={self._beta:g} eta={self.eta:g} "
            f"gain={self.server_gain:g} k={cfg.k}"
        )

    @property
    def eta(self) -> float:
        """Server learning rate: ``cfg.server_lr``, or the gain-neutral ``1 - beta``."""
        if self.cfg.server_lr is not None:
            return float(self.cfg.server_lr)
        return 1.0 - self._beta

    @property
    def server_gain(self) -> float:
        return self.eta / (1.0 - self._beta)

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

    def drift_norm(self, net: torch.nn.Module | None = None) -> float:
        """L2 distance of the federated tensors from the current round's anchor.

        This is the quantity the proximal term is built on: its gradient is
        ``mu * (w - anchor)``, so a norm that stays near zero means the leash has
        nothing to restrain (which is what aggregating after every update does).
        """
        net = self.network if net is None else net
        total = torch.zeros((), device=hardware.device)
        for name, param in net.named_parameters():
            anchor = self._global_params.get(name)
            if anchor is None:
                continue
            total = total + torch.sum((param.detach() - anchor) ** 2)
        return float(total.sqrt().item())

    def sharing_report(self) -> dict:
        """Checksum and magnitude of the federated tensors as this rank holds them.

        Two ranks that hold the same shared model report the same pair; the
        magnitudes stay comparable even when float summation order makes the
        hashes differ by a bit.
        """
        digest = sha1()
        magnitude = 0.0
        for name, tensor in sorted(self._synced_state().items()):
            digest.update(name.encode())
            digest.update(tensor.numpy().tobytes())
            magnitude += float(tensor.abs().sum())
        return {"sha1": digest.hexdigest(), "l1": magnitude}

    def log_sharing(self) -> None:
        """Show in the log that the round really left one shared model.

        The all-reduce cannot demonstrate this by itself: it would also "work" if
        the ranks kept their own copies afterwards. Comparing what every rank
        actually holds turns the sharing assumption into something the log states
        (and would catch a rank silently missing from a round).
        """
        if pdist.world_size() < 2:
            return
        reports = pdist.gather_objects(self.sharing_report())
        if not pdist.is_main():
            return
        scale = max(abs(r["l1"]) for r in reports) or 1.0
        spread = max(r["l1"] for r in reports) - min(r["l1"] for r in reports)
        if len({r["sha1"] for r in reports}) == 1:
            logger.info(
                f"Round {self._rounds:4d} | one shared model: all "
                f"{len(reports)} ranks hold sha1={reports[0]['sha1'][:12]}"
            )
        elif spread / scale < 1e-9:
            logger.info(
                f"Round {self._rounds:4d} | shared model identical to float noise "
                f"(relative L1 spread {spread / scale:.2e})"
            )
        else:
            logger.warning(
                f"Round {self._rounds:4d} | ranks hold DIFFERENT federated models "
                f"(relative L1 spread {spread / scale:.2e})"
            )

    def _global_path(self) -> Path | None:
        """Where the shared federated checkpoint lives, for every client to read."""
        directory = type(self).shared_dir(self.cfg)
        if directory is None:
            return None
        return directory / f"ckp_{self.current_update}.pt"

    def _save_global(self) -> None:
        """Rank 0 writes the aggregated model where all clients can find it.

        Only on aggregation rounds (between them the local weights are this
        client's own, and writing those as "the federated model" would hand every
        peer a single client's model on resume), and at most every
        ``save_interval`` updates — a save turn has to fall on a round boundary,
        which with K > 1 it usually does not.
        """
        if not self._aggregated or not pdist.is_main():
            return
        interval = max(1, self.cfg.save_interval)
        if self.current_update - self._shared_saved_at < interval:
            return
        filepath = self._global_path()
        if filepath is None:
            return
        self._shared_saved_at = self.current_update
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = self._checkpoint()
        # a peer's personal copy (Ditto) is not the server's to hand out
        payload.pop("personal_network", None)
        payload.pop("personal_optimizer", None)
        torch.save(payload, filepath)
        logger.info(f"Saved federated checkpoint to {filepath}")

    def _reset_optimizer_state(self) -> None:
        """Drop Adam's moments at a round boundary, so each round starts fresh."""
        for pg in self.optim.param_groups:
            for p in pg.get("params", []):
                state = self.optim.state.get(p)
                if state is not None and "exp_avg" in state:
                    state["exp_avg"].zero_()
                    state["exp_avg_sq"].zero_()

    def sync(self) -> None:
        """Called after every local update; aggregates every ``sync_every`` of them.

        With ``sync_every = 1`` the client is re-anchored after each update, so
        drift never accumulates and the proximal term stays inert. Larger values
        let it walk for several updates before the server sees it — the regime
        FedProx is designed for (the K = 3 / K = 7 ablations). The rollout policy
        refresh and artifact saving stay per update either way, so only the
        aggregation and the anchor move to the round boundary.
        """
        every = max(1, self.cfg.k)
        self._rounds += 1
        self._aggregated = self._rounds % every == 0
        if self._aggregated:
            # a lone client is its own aggregate, so there is nothing to reduce
            if pdist.world_size() > 1:
                self._aggregate_distributed()
            # the anchor the next round measures against
            self._global_params = self._global_snapshot()
            self._reset_optimizer_state()
            if pdist.world_size() > 1:
                # every rank logs its own client; this one is the mean over them
                self.federation_log()
                # and this one proves the ranks really hold the same model now
                self.log_sharing()
            # after the collectives, so rank 0's disk write delays nobody
            self._save_global()
        self._sync_inference()
        self._periodic_save()

    def _aggregate_distributed(self) -> None:
        """FedAvgM over ranks: averaged weights plus momentum on the update.

        The momentum is stored as an EMA of the deltas (``m = beta*m + (1-beta)*d``)
        and the applied step is scaled by ``server_gain``, which is exactly
        FedAvgM's ``v = beta*v + d`` with server learning rate ``eta``:

            w += eta * v   ==   w += (eta / (1 - beta)) * m

        The first round seeds the buffer with the observed delta, i.e. ``v = d``,
        which is the accumulator's own initialisation: at ``server_lr = 1`` this is
        the paper's scheme verbatim, and at the default it makes round 1 a full
        FedAvg step instead of ``(1 - beta)`` of one (the same effect as the bias
        correction on Adam's first step). Round 1 and the first round after a
        resume therefore differ from a textbook EMA by a factor ``1 / (1 - beta)``.
        """
        avg = pdist.mean_tensors(self._synced_state())
        delta = {k: avg[k] - self._global[k].to(dtype=avg[k].dtype) for k in avg}
        if not self._momentum:
            self._momentum = {k: v.clone() for k, v in delta.items()}
        else:
            for k, v in delta.items():
                self._momentum[k] = (
                    self._beta * self._momentum[k] + (1.0 - self._beta) * v
                )
        gain = self.server_gain
        self._global = {
            k: self._global[k].to(dtype=avg[k].dtype) + gain * self._momentum[k]
            for k in avg
        }
        self._version += 1
        self._load_global(self._global)
