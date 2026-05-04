from dataclasses import dataclass

import numpy as np
import torch

from calvin_env_modified.envs.observation import CalvinEnvObservation
from heca.environment.converters.converter import ObsConverter
from heca.misc.td import TDEntities


class CalvinHecaConverterImage(ObsConverter):
    @dataclass(kw_only=True)
    class Config(ObsConverter.Config):
        label: str = "v1"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, obs: CalvinEnvObservation) -> TDEntities:
        obs.rgb
