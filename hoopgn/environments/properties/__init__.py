from hoopgn.environments.properties.property import Property, PropertyConfig


def select_properties(configs: list[PropertyConfig]) -> list[Property]:
    """Create states from configs - simple factory function"""
    return [Property(config) for config in configs]
