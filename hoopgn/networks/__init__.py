from hoopgn.networks.baseline import BaselineNetwork, BaselineNetworkConfig
from hoopgn.networks.v1 import HoopgnV1Network, HoopgnV1Config
from hoopgn.networks.network import Network, NetworkConfig


def select_network(config: NetworkConfig) -> Network:
    """Create network from config - simple factory function"""
    if isinstance(config, BaselineNetworkConfig):
        return BaselineNetwork(config)
    elif isinstance(config, HoopgnV1Config):
        return HoopgnV1Network(config)
    else:
        raise ValueError(f"Unknown network type: {type(config)}")
