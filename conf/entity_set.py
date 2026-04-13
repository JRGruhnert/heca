from hoopgn.entities.entity import EntityConfig


def get_entity_set(tag: str) -> list[EntityConfig]:
    return _ENTITY_SETS[tag]  # type: ignore
