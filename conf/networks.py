"""Named network configurations.

Select one in the training pipeline with ``--network <name>``.
"""

from heca.heca_gnn.network import Network
from heca.heca_gnn.network2 import Network2

# Network2 (option memory + interaction) variants.
default = Network2.Config()
small = Network2.Config(feature_dim=128, encoder_depth=2, gnn_mlp_depth=2)
big = Network2.Config(feature_dim=512, encoder_depth=4, gnn_mlp_depth=4, attn_heads=8)

# Plain Network variants.
base = Network.Config()

NETWORK_NAMES = ["default", "small", "big", "base"]
