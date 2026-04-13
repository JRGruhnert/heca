import numpy as np
from tensordict import TensorDict
import torch

from hoopgn.observation import empty_batchsize


class TDParameters(TensorDict):
    def __init__(self, parameters: dict[str, torch.Tensor]):
        super().__init__(parameters, batch_size=empty_batchsize)

    @classmethod
    def from_numpy_dict(cls, data: dict[str, np.ndarray]) -> "TDParameters":
        return cls({k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()})

    def same_fields(self, other: "TDParameters") -> bool:
        return set(self.keys()) == set(other.keys())  # type: ignore

    def equal(self, other: "TDParameters") -> bool:
        result = self == other
        if isinstance(result, bool):
            return result
        return bool(result.all())

    def get_parameter(self, key: str) -> torch.Tensor:
        if key not in self.keys():
            raise KeyError()
        return self[key]
