from typing import Protocol, cast

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage

MemoryStep = tuple[torch.Tensor, torch.Tensor]


class RowStore(Protocol):
    """Features every entity-ish row carries."""

    x: torch.Tensor
    type_ids: torch.Tensor


class EntityRowStore(RowStore, Protocol):
    role_ids: torch.Tensor


class CompRowStore(RowStore, Protocol):
    weight: torch.Tensor


class CanonicalRowStore(RowStore, Protocol):
    role_ids: torch.Tensor
    cur_idx: torch.Tensor
    goal_idx: torch.Tensor


class OptionStore(Protocol):
    x: torch.Tensor
    # 1.0 for options the gate rejects for the current start, 0.0 otherwise:
    # the export keeps every option and carries feasibility as data.
    gated: torch.Tensor


class StateStore(Protocol):
    x: torch.Tensor
    type_ids: torch.Tensor
    budget: torch.Tensor


class HecaData(HeteroData):
    _global_store: BaseStorage

    @property
    def mem_step(self) -> MemoryStep | None:
        return cast(MemoryStep | None, self._global_store.get("mem_step"))

    @mem_step.setter
    def mem_step(self, value: MemoryStep | None) -> None:
        self._global_store["mem_step"] = value

    @property
    def entity(self) -> EntityRowStore:
        return cast(EntityRowStore, self["entity"])

    @property
    def comp(self) -> CompRowStore:
        return cast(CompRowStore, self["comp"])

    @property
    def canonical(self) -> CanonicalRowStore:
        return cast(CanonicalRowStore, self["canonical"])

    @property
    def option(self) -> OptionStore:
        return cast(OptionStore, self["option"])

    @property
    def state(self) -> StateStore:
        return cast(StateStore, self["state"])
