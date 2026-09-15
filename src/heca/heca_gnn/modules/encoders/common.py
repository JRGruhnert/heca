import torch
from torch import nn

from heca.data.entity import Entity
from heca.data.free import FreeEntity
from heca.data.prismatic import PrismaticEntity
from heca.data.revolute import RevoluteEntity
from heca.data.static import StaticEntity
from heca.data.entity import Entity


class _Block(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim // 2),
            nn.GELU(),
            nn.Linear(out_dim // 2, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _EntityEncoder(nn.Module):
    BLOCKS: tuple[str, ...] = ()

    def __init__(self, out_dim: int, rotation: bool = True, logstd: bool = True):
        super().__init__()
        self.logstd = logstd
        self.out_dim = out_dim
        layout = Entity.LAYOUT if logstd else Entity.POINT_LAYOUT
        blocks = tuple(b for b in self.BLOCKS if rotation or b != "rot")
        if not blocks:
            raise ValueError(f"{type(self).__name__} has no blocks left")
        self.blocks = blocks

        n_blocks = len(Entity.LAYOUT)
        share = out_dim // n_blocks
        self.block_widths = {
            name: (share if i < n_blocks - 1 else out_dim - share * (n_blocks - 1))
            for i, name in enumerate(Entity.LAYOUT)
        }
        self.subs = nn.ModuleDict(
            {name: _Block(layout[name].dim, self.block_widths[name]) for name in blocks}
        )

    def _slices(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        layout = Entity.LAYOUT if self.logstd else Entity.POINT_LAYOUT
        out: dict[str, torch.Tensor] = {}
        inv = 1.0 / abs(Entity.BASE_LOGSTD)
        for name in self.blocks:
            block = layout[name]
            parts = [x[:, block.mean()]]
            if block.logstd_dim:
                parts.append(x[:, block.logstd()] * inv)
            out[name] = parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = self._slices(x)
        out = x.new_zeros(x.shape[0], self.out_dim)
        offset = 0
        for name in Entity.LAYOUT:  # canonical order: state, pos, rot, extra
            width = self.block_widths[name]
            if name in self.blocks:
                out[:, offset : offset + width] = self.subs[name](parts[name])
            offset += width
        return out


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


class SetEncoder(nn.Module):
    def __init__(self, dim: int, rotation: bool = True, logstd: bool = True):
        nn.Module.__init__(self)
        self.dim = dim

        self.encoders = nn.ModuleDict(
            {name: cls(dim, rotation, logstd) for name, cls in self.encoder_map.items()}
        )

    @property
    def encoder_map(self) -> dict[str, type[_EntityEncoder]]:
        return {
            "free": FreeEncoder,
            "static": StaticEncoder,
            "prismatic": PrismaticEncoder,
            "revolute": RevoluteEncoder,
        }

    def encode(self, x: torch.Tensor, type_ids: torch.Tensor) -> torch.Tensor:
        """Encode rows by entity type: ``x`` is (n, FEATURE_DIM), ids are (n,)."""
        out = x.new_zeros(x.shape[0], self.dim)
        for t, name in enumerate(self.encoder_map):
            rows = type_ids == t
            if rows.any():
                out[rows] = self.encoders[name](x[rows])
        return out
