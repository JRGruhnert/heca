from dataclasses import dataclass
from abc import abstractmethod
from tensordict import TensorDict
from torch import nn, Tensor
from torch_geometric.data import Batch
from hoopgn.misc.classes import StoragableClass
from hoopgn.misc.td import TDScene
from hoopgn.entities.properties.encoders.encoder import PropertyEncoder


class Hoop(StoragableClass, nn.Module):
    @dataclass(kw_only=True)
    class Query(StoragableClass.Query):
        label: str

    @dataclass(kw_only=True)
    class Config(StoragableClass.Config):
        eval_mode: bool = False
        dim_encoder: int = 32

    def __init__(self, cfg: Config):
        nn.Module.__init__(self)
        self.cfg = cfg

    def recursive_encode(self, tensor_dict: TDScene, prefix=""):
        for key, value in tensor_dict.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, TensorDict):
                self.recursive_encode(value, prefix=full_key)
            else:
                query = PropertyEncoder.Query(label=full_key)  # Validate existence
                encoder = PropertyEncoder.search(query)
                tensor_dict[key] = encoder(value)

    def encode(self, x: TDScene, y: TDScene) -> tuple[TDScene, TDScene]:
        self.recursive_encode(x)
        self.recursive_encode(y)
        return x, y

    @abstractmethod
    def forward(self, x: Batch) -> tuple[Tensor, Tensor]:
        raise NotImplementedError()

    @abstractmethod
    def explain(self, batch: Batch) -> tuple:
        raise NotImplementedError("This network does not support explanations.")
