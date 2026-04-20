from abc import abstractmethod
from dataclasses import dataclass

import torch
from hoopgn.environments.properties.features.feature import PropertyFeature


class PropertyParameter(PropertyFeature):
    @dataclass(kw_only=True)
    class Config(PropertyFeature.Config):
        pass

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    # TODO: This whole file is just there for legacy code.
    # Remove it as soon as starting with master
    def __call__(
        self,
        value: tuple | torch.Tensor | list[float] | float | int | None,
    ) -> torch.Tensor:
        if isinstance(value, torch.Tensor):
            return value
        else:
            if isinstance(value, (float, int)):
                return torch.tensor([value], dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported value type: {type(value)}")

    @abstractmethod
    def hoopgnv1(
        self,
        start: torch.Tensor,
        end: torch.Tensor,
        reversed: bool,
        selected_by_tapas: bool = False,
    ) -> torch.Tensor | None:
        raise NotImplementedError("Subclasses must implement this method.")
