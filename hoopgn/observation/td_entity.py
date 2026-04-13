from tensordict import TensorDict
import torch

from hoopgn.observation import empty_batchsize


class TDEntity(TensorDict):
    def __init__(
        self,
        position: torch.Tensor,
        rotation: torch.Tensor,
        domain: torch.Tensor,
        state: torch.Tensor,
    ):
        data = {
            "position": position,
            "rotation": rotation,
            "domain": domain,
            "state": state,
        }
        super().__init__(data, batch_size=empty_batchsize)

    @property
    def position(self) -> torch.Tensor:
        return self["position"]

    @property
    def rotation(self) -> torch.Tensor:
        return self["rotation"]

    @property
    def domain(self) -> torch.Tensor:
        return self["domain"]

    @property
    def state(self) -> torch.Tensor:
        return self["state"]
