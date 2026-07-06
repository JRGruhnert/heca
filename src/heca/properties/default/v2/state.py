from dataclasses import dataclass
import numpy as np
import torch

from heca.properties.encoders.encoder import PropertyEncoder
from heca.properties.encoders.v2.state import StateEncoder
from heca.properties.evaluators.state import StateEvaluator
from heca.properties.evaluators.evaluator import PropertyEvaluator
from heca.properties.rulers.state import StateRuler
from heca.properties.rulers.ruler import PropertyRuler
from heca.properties.default.v2.property import Property


class StateProperty(Property):
    @dataclass(kw_only=True)
    class Config(Property.Config):
        ruler: PropertyRuler.Config = StateRuler.Config()
        encoder: PropertyEncoder.Config = StateEncoder.Config()
        evaluator: PropertyEvaluator.Config = StateEvaluator.Config()
        values: set[str]

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        assert len(cfg.values) > 0, "State must have at least one value."
        assert None not in cfg.values, "State values cannot be None."

        self.one_hots: dict[str, torch.Tensor] = {
            label: self.make_one_hot(label) for label in cfg.values
        }
        self.zeros = self.make_zeros()

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.values), dtype=torch.float32)

    def make_zeros_dc(self) -> np.ndarray:
        return np.zeros(len(self.cfg.values))

    def make_one_hot(self, label: str) -> torch.Tensor:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.values, "Label must be in state values."
        one_hot = self.make_zeros()
        index = list(self.cfg.values).index(label)
        one_hot[index] = 1.0
        return one_hot

    def one_hot_from_idx(self, idx: int) -> torch.Tensor:
        assert 0 <= idx < len(self.cfg.values), "Index out of bounds."
        one_hot = self.make_zeros()
        one_hot[idx] = 1.0
        return one_hot

    def one_hot_from_idx_dc(self, idx: int) -> np.ndarray:
        assert 0 <= idx < len(self.cfg.values), "Index out of bounds."
        one_hot = self.make_zeros_dc()
        one_hot[idx] = 1.0
        return one_hot
