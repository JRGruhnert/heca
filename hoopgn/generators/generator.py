from dataclasses import dataclass

from hoopgn.agents.agent import Agent
from hoopgn.entities.entity import Entity
from hoopgn.misc.classes import ConfigClass
from hoopgn.misc.td import TDScene
from torch_geometric.data import Batch, HeteroData


class Generator(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        pass

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, Entity]], HeteroData]:
        raise NotImplementedError()

    def to_batch(self, data: list[HeteroData]) -> Batch:
        return cast(Batch, Batch.from_data_list(data))  # type: ignore
