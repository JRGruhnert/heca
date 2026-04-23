from dataclasses import dataclass

from calvin_env_modified.envs.observation import CalvinEnvObservation
from hoopgn.observation.converters.converter import Converter
from hoopgn.observation.td_entities import TDEntities


class V2Converter(Converter):
    @dataclass(kw_only=True)
    class Config(Converter.Config):
        label: str = "v2"

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def __call__(self, obs: CalvinEnvObservation) -> TDEntities:
        return TDEntities({})
