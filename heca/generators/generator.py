from dataclasses import dataclass

from heca.agents.agent import Agent
from heca.misc.base import Configurable
from heca.misc.td import TDScene
from torch_geometric.data import Batch, HeteroData


class HecaGenerator(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agents: set[Agent.Config]

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def reset(self, x: TDScene, y: TDScene) -> HeteroData:
        raise NotImplementedError

    def step(
        self, x: TDScene
    ) -> tuple[list[tuple[Agent.Config, TDScene, TDScene]], HeteroData]:
        raise NotImplementedError()

    def to_batch(self, data: list[HeteroData]) -> Batch:
        return cast(Batch, Batch.from_data_list(data))  # type: ignore
