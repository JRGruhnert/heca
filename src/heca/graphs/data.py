from collections.abc import Generator
from contextlib import contextmanager
from typing import Protocol, cast

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage

TrunkMemory = dict[str, torch.Tensor]


class RowStore(Protocol):
    x: torch.Tensor
    type_ids: torch.Tensor


class EntityRowStore(RowStore, Protocol):
    role_ids: torch.Tensor


class CompRowStore(RowStore, Protocol):
    weight: torch.Tensor


class OptionStore(Protocol):
    x: torch.Tensor
    gated: torch.Tensor


class OptionStateStore(Protocol):
    x: torch.Tensor
    type_ids: torch.Tensor
    gated: torch.Tensor


class HecaData(HeteroData):
    _global_store: BaseStorage

    @property
    def memory(self) -> TrunkMemory:
        return cast(TrunkMemory, self._global_store.get("memory") or {})

    @memory.setter
    def memory(self, value: TrunkMemory) -> None:
        self._global_store["memory"] = dict(value)

    @property
    def budget(self) -> torch.Tensor:
        return cast(torch.Tensor, self._global_store["budget"])

    @property
    def entity(self) -> EntityRowStore:
        return cast(EntityRowStore, self["entity"])

    @property
    def comp(self) -> CompRowStore:
        return cast(CompRowStore, self["comp"])

    @property
    def option(self) -> OptionStore:
        return cast(OptionStore, self["option"])

    @property
    def option_state(self) -> OptionStateStore:
        return cast(OptionStateStore, self["option_state"])


@contextmanager
def installed_memory(
    data: HecaData, memory: TrunkMemory
) -> Generator[None, None, None]:
    stored = data.memory
    data.memory = memory
    try:
        yield
    finally:
        data.memory = stored
