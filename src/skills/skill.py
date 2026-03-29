from dataclasses import dataclass
from functools import cached_property
import torch
import numpy as np
from src.observation.observation import StateValueDict
from src.states.logic.condition import Condition, ConditionConfig


@dataclass
class Skill2Config:
    label: str
    id: int


class Skill2:
    def __init__(
        self,
        config: Skill2Config,
    ):
        self.config = config
        self.precons: dict[str, Condition] = {}
        self.postcons: dict[str, Condition] = {}
