import torch
from torch import nn

from heca.graphs.data import HecaData
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.data.free import FreeEntity
from heca.data.entity import Entity


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EntityEncoder(nn.Module):
    BLOCKS: tuple[str, ...] = ()

    def __init__(self, out_dim: int):
        super().__init__()
        if "state" not in self.BLOCKS:
            raise ValueError("every entity has a state block")

        # One slice of the embedding per block, sized by the block's share.
        share = max(out_dim // len(self.BLOCKS), 8)
        dims = {name: share for name in self.BLOCKS}
        dims[self.BLOCKS[-1]] = out_dim - share * (len(self.BLOCKS) - 1)

        self.dims = dims
        self.subs = nn.ModuleDict(
            {
                name: (
                    nn.Linear(Entity.LAYOUT[name].dim, dims[name], bias=False)
                    if name == "state"
                    else _Block(Entity.LAYOUT[name].dim, dims[name])
                )
                for name in self.BLOCKS
            }
        )

    def _slices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        out: dict[str, torch.Tensor] = {}
        inv = 1.0 / abs(Entity.BASE_LOGSTD)
        for name in self.BLOCKS:
            block = Entity.LAYOUT[name]
            parts = [x[:, block.mean()]]
            if block.logstd_dim:
                parts.append(x[:, block.logstd()] * inv)
            out[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._slices(x)
        return torch.cat([self.subs[name](parts[name]) for name in self.BLOCKS], dim=-1)


class FreeEncoder(_EntityEncoder):
    """Free body: position and orientation."""

    BLOCKS = FreeEntity.BLOCKS


class StaticEncoder(_EntityEncoder):
    """Static prop: position only."""

    BLOCKS = StaticEntity.BLOCKS


class PrismaticEncoder(_EntityEncoder):
    """Prismatic joint: position, orientation and the slide offset."""

    BLOCKS = PrismaticEntity.BLOCKS


class RevoluteEncoder(_EntityEncoder):
    """Revolute joint: position, orientation and the joint angle."""

    BLOCKS = RevoluteEntity.BLOCKS


class EntityRowEncoder(nn.Module):
    def __init__(self, dim: int):
        nn.Module.__init__(self)
        if tuple(self.encoder_map) != Entity.TYPE_NAMES:
            raise ValueError(
                f"encoder_map order {tuple(self.encoder_map)} does not match "
                f"Entity.TYPE_NAMES {Entity.TYPE_NAMES}"
            )
        self.dim = dim
        self.entity_encoders = nn.ModuleDict(
            {name: cls(dim) for name, cls in self.encoder_map.items()}
        )

        self.comp_encoders = nn.ModuleDict(
            {name: cls(dim) for name, cls in self.encoder_map.items()}
        )

    @property
    def encoder_map(self) -> dict[str, type[_EntityEncoder]]:
        return {
            "free": FreeEncoder,
            "static": StaticEncoder,
            "prismatic": PrismaticEncoder,
            "revolute": RevoluteEncoder,
        }

    def encode(self, node_type: str, data: HecaData) -> torch.Tensor:
        encoders = self.comp_encoders if node_type == "comp" else self.entity_encoders
        x = data[node_type].x
        type_ids = data[node_type].type_ids
        out = x.new_zeros(x.shape[0], self.dim)
        for t, name in enumerate(self.encoder_map):
            rows = type_ids == t
            if rows.any():
                out[rows] = encoders[name](x[rows])
        return out
