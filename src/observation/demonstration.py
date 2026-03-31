import torch


class Demo:
    def __init__(self):
        self.precons_raw: dict[str, torch.Tensor] = {}
        self.postcons_raw: dict[str, torch.Tensor] = {}

    def _load_precons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement method.")

    def _load_postcons(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError("Subclasses must implement method.")
