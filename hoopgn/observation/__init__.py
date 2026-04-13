import torch

from hoopgn.entities.entity import Entity
from hoopgn.observation.td_entities import TDEntities
from hoopgn.observation.td_parameters import TDParameters

empty_batchsize = torch.Size([])


def convert_parameters_to_entities(
    current: TDParameters, goal: TDParameters
) -> tuple[list[Entity], TDEntities, TDEntities]:
    # TODO: step to convert v1 to v2 hoopgn, should be removed in the future
    return [Entity()], TDEntities(**current.__dict__), TDEntities(**goal.__dict__)
