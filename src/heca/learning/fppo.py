from dataclasses import dataclass
from pathlib import Path

import torch

from heca.learning import dist as pdist
from heca.learning.optim import ClientAdamW
from heca.learning.ppo import PPO
from heca.misc import hardware, logger


class FPPO(PPO):
    @dataclass(kw_only=True)
    class Config(PPO.Config):
        fedadamw_alpha: float | None = None
        k: int = 1

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self._sync_keys = self.network.sync_keys(cfg.network.sync)
        self._use_fedadamw = cfg.fedadamw_alpha is not None
        self._alpha = float(cfg.fedadamw_alpha or 0.0)
        self._v_bar: dict[str, torch.Tensor] = {}
        self._applied: dict[str, torch.Tensor] = {}
        self._local_steps = 0
        self._previous_steps = 0
        self._rounds = 0
        self._aggregated = False
        self._shared_saved_at = 0
        # rank 0's weights are the initial global model for every rank
        self._global = self._synced_state()
        pdist.broadcast_tensors(self._global)
        self._load_global(self._global)
        self._global_params = self._global_snapshot()

    def _make_optimizer(self) -> torch.optim.Optimizer:
        return ClientAdamW(
            self.network, lr=self.cfg.lr, weight_decay=self.cfg.weight_decay
        )

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
        """L2 distance of the federated tensors from the current round's anchor."""
        net = self.network if net is None else net
        total = torch.zeros((), device=hardware.device)
        for name, param in net.named_parameters():
            anchor = self._global_params.get(name)
            if anchor is None:
                continue
            total = total + torch.sum((param.detach() - anchor) ** 2)
        return float(total.sqrt().item())

    def _global_path(self) -> Path | None:
        """Where the shared federated checkpoint lives, for every client to read."""
        directory = type(self).shared_dir(self.cfg)
        if directory is None:
            return None
        return directory / f"ckp_{self.current_update}.pt"

    def _save_global(self) -> None:
        """Rank 0 writes the aggregated model where all clients can find it."""
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

    def _second_moments(self) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        for name, param in self.network.named_parameters():
            if name not in self._sync_keys:
                continue
            state = self.optim.state.get(param)
            if state is not None and "exp_avg_sq" in state:
                # gloo reduces on CPU, and the weights are moved there too
                out[name] = (
                    state["exp_avg_sq"]
                    .detach()
                    .to(device="cpu", dtype=torch.float32, copy=True)
                )
            else:
                shared = self._v_bar.get(name)
                out[name] = (
                    torch.zeros(param.shape, dtype=torch.float32)
                    if shared is None
                    else shared.detach().to(
                        device="cpu", dtype=torch.float32, copy=True
                    )
                )
        return out

    def _optim_update_hook(self) -> None:
        self._local_steps += 1
        if not self._use_fedadamw or not self._applied or not self._previous_steps:
            return
        scale = self._alpha / self._previous_steps
        with torch.no_grad():
            for name, param in self.network.named_parameters():
                delta = self._applied.get(name)
                if delta is None:
                    continue
                param.add_(delta.to(dtype=param.dtype) * scale)

    def sync(self) -> None:
        every = max(1, self.cfg.k)
        self._rounds += 1
        self._aggregated = self._rounds % every == 0
        if self._aggregated:
            if pdist.world_size() > 1:
                self._aggregate_distributed()
            # the anchor the next round measures against
            self._global_params = self._global_snapshot()
            self._previous_steps = max(1, self._local_steps) if self._local_steps else 0
            self._local_steps = 0
            assert isinstance(self.optim, ClientAdamW)
            self.optim.start_round(self._v_bar if self._use_fedadamw else None)
            if pdist.world_size() > 1:
                self.federation_log()
            self._save_global()
        self._sync_inference()
        self._periodic_save()

    def _aggregate_distributed(self) -> None:
        """FedAvg over ranks: the averaged weights are the new global model."""
        avg = pdist.mean_tensors(self._synced_state())
        step = {k: avg[k] - self._global[k].to(dtype=avg[k].dtype) for k in avg}
        if self._use_fedadamw:
            moments = self._second_moments()
            shared = pdist.mean_tensors(moments) if moments else {}
            self._v_bar = {
                name: tensor.to(device="cpu", dtype=torch.float32)
                for name, tensor in shared.items()
            }
        self._applied = {k: v.detach().clone() for k, v in step.items()}
        self._global = {
            k: self._global[k].to(dtype=avg[k].dtype) + step[k] for k in avg
        }
        self._load_global(self._global)
