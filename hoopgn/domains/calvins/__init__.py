def properties_to_entities(properties: list[Property.Config]) -> list[Entity.Config]:
    property_dict = {prop.label: prop for prop in properties}
    entities: list[Entity.Config] = []
    for entity_name, props in _ENTITY_PROPERTY_MAPPING.items():
        if (
            props["position"] not in property_dict
            or props["rotation"] not in property_dict
            or props["scalar"] not in property_dict
        ):
            continue

        cfg = Entity.Config(
            id=len(entities),
            label=entity_name,
            position=property_dict[props["position"]],
            rotation=property_dict[props["rotation"]],
            state=property_dict[props["scalar"]],
        )
        entities.append(cfg)
    return entities


def tdp_to_tde(properties: TDProperties, entity: Entity) -> TDEntity:
    return TDEntity(
        position=properties[entity.cfg.position.label],
        rotation=properties[entity.cfg.rotation.label],
        state=properties[entity.cfg.state.label],
    )
