from hoopgn.entities.entity import Entity, EntityConfig


def select_entities(configs: list[EntityConfig]) -> list[Entity]:
    """Create entities from configs - simple factory function"""
    return [Entity(config) for config in configs]
