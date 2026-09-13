from contextlib import contextmanager
from typing import Iterator, Protocol, cast

import torch
from torch_geometric.data import HeteroData
from torch_geometric.data.storage import BaseStorage


# trunk name -> the memory that decision was conditioned on (missing = none yet)
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
        """Recurrent memory this decision is conditioned on, per trunk.

        Keyed by trunk name (``Network.memory_keys``), so a network with separate
        actor/critic trunks keeps two independent recurrences and a shared trunk
        keeps one; a missing key means "no memory yet", which is the normal state
        of the first decision of an episode (the network materialises zeros).

        This is the *only* way memory enters a forward pass - the trainer
        overwrites it while replaying a chunk (see :func:`installed_memory`).

        NOTE: ``pyg`` writes attributes straight into the store and therefore
        bypasses the setter below, so the mapping is kept *by reference*: hand
        over a freshly built dict, and do not mutate one that is already stored
        in a buffered transition.
        """
        return cast(TrunkMemory, self._global_store.get("memory") or {})

    @memory.setter
    def memory(self, value: TrunkMemory) -> None:
        self._global_store["memory"] = dict(value)

    @property
    def budget(self) -> torch.Tensor:
        """Option budget left in the episode: one value per decision."""
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
def installed_memory(data: HecaData, memory: TrunkMemory) -> Iterator[None]:
    """Condition ``data`` on ``memory`` (per trunk) for the duration of the block.

    Used by truncated-BPTT replay: the first transition of a chunk is scored
    with the memory recorded in the buffer (detached), every later one with the
    live tensors produced by the previous step, so the GRUs get gradient across
    the chunk. The recorded mapping is restored afterwards, because the buffer is
    replayed once per epoch.
    """
    stored = data.memory
    data.memory = memory
    try:
        yield
    finally:
        data.memory = stored
