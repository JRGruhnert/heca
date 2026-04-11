from dataclasses import dataclass


@dataclass
class ObjectDeltaPoint:
    position_delta: float
    rotation_delta: float
    state: int
    cluster: int


@dataclass
class ObjectLocationPoint:
    x: float
    y: float
    z: float
    rotation: float
    state: int
