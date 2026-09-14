from torch import nn


def _make_gnn_mlp(
    dim: int, num_layers: int, in_dim: int | None = None
) -> nn.Sequential:
    """MLP used by the graph blocks, optionally reading a wider input (e.g. a
    concatenated src/dst pair) while writing ``dim``."""
    layers = [nn.Linear(in_dim or dim, dim), nn.LayerNorm(dim), nn.ReLU()]
    for _ in range(num_layers - 1):
        layers.append(nn.Linear(dim, dim))
        layers.append(nn.LayerNorm(dim))
        layers.append(nn.ReLU())
    layers.append(nn.Linear(dim, dim))
    return nn.Sequential(*layers)
