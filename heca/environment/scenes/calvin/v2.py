from heca.entities.entity import Entity

entities = [
    Entity.Config(
        env="calvin",
        label="ee",
        states={"open", "closed"},
    ),
    Entity.Config(
        env="calvin",
        label="drawer",
        states={"open", "closed"},
    ),
    Entity.Config(
        env="calvin",
        label="slider",
        states={"open", "closed", "half-open"},
    ),
    Entity.Config(
        env="calvin",
        label="button",
        states={"pressed", "released"},
    ),
    # Entity.Config(
    #    env="calvin",
    #    label="switch",
    #    states={"on", "off"},
    # ),
    Entity.Config(
        env="calvin",
        label="light",
        states={"on", "off"},
    ),
    Entity.Config(
        env="calvin",
        label="red_block",
        states={"grabbed", "ungrabbed"},
    ),
    Entity.Config(
        env="calvin",
        label="pink_block",
        states={"grabbed", "ungrabbed"},
    ),
    Entity.Config(
        env="calvin",
        label="blue_block",
        states={"grabbed", "ungrabbed"},
    ),
]
