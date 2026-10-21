from dataclasses import dataclass
from typing import ClassVar

import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class Reference:
    image: Image.Image
    x: int
    y: int
    xyz: np.ndarray
    frame: int | None = None
    POSITION: ClassVar[str] = "pos"


@dataclass(frozen=True, slots=True)
class Keypoint:
    xyz: np.ndarray  # world position, raw (unnormalised) metres
    score: float  # the matcher's confidence for this point
    x: int  # pixel column in that image
    y: int  # pixel row in that image
