from dataclasses import dataclass

from hoopgn.properties.features.extractors.calvin_gt_extractor import (
    CalvinGTExtractor,
)

from hoopgn.properties.property import Property
from hoopgn.properties.rotations.quaternion import Quaternion


@dataclass
class RotationConfig(Property.Config):
    extractor: CalvinGTExtractor.Config

    encoder = Quaternion.encoder
    normalizer = Quaternion.normalizer
    ruler = Quaternion.ruler
    evaluator = Quaternion.evaluator
    condition = Quaternion.condition
