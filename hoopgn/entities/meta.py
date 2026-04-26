from hoopgn.entities.entity import Entity
from hoopgn.entities.real import RealEntity
from hoopgn.environments.environment import Environment
from hoopgn.misc.td import TDWorld
from hoopgn.properties.property import Property


class MetaEntity(Entity):
    @classmethod
    def ensure_same_meta(cls, entities: list[Entity]) -> str:
        if len(set(e.Config.meta for e in entities)) > 1:
            raise ValueError("All entities must have the same meta")
        return entities[0].Config.meta

    @classmethod
    def ensure_same_version(cls, entities: list[Entity]) -> str:
        if len(set(e.cfg.version for e in entities)) > 1:
            raise ValueError("All entities must have the same environment version")
        return entities[0].cfg.version

    @classmethod
    def ensure_same_env(cls, entities: list[RealEntity]) -> Environment.Query:
        if len(set(e.Config.env for e in entities)) > 1:
            raise ValueError("All entities must have the same environment")
        return entities[0].Config.env

    @classmethod
    def make_properties(cls, entities: list[Entity]) -> list[Property.Config]:
        raise NotImplementedError()

    @classmethod
    def create(
        cls, meta: str, version: str, props: list[Property.Config], label: str
    ) -> Entity:
        return Entity(
            cfg=cls.Config(
                label=label,
                meta=meta,
                version=version,
                properties=props,
            )
        )

    @classmethod
    def ensure_basics(cls, entities: list[Entity]) -> tuple[str, str]:
        meta = cls.ensure_same_meta(entities)
        version = cls.ensure_same_version(entities)
        # env = cls.ensure_same_env(entities)
        return meta, version

    @classmethod
    def merge(cls, entities: list[Entity], cluster: str, label: str) -> Entity:
        old, version = cls.ensure_basics(entities)
        props = cls.make_properties(entities)
        meta = f"{old}.{cluster}"
        return cls.create(meta=meta, version=version, props=props, label=label)

    @classmethod
    def distance(cls, a: TDWorld, b: TDWorld, e: Entity) -> float:
        cls.ensure_basics([a, b])
        values = []
        for k, v in e.properties.items():
            if k not in a or k not in b:
                raise KeyError()
            d = v[0].distance(a[k], b[k])
            values.append(d * v[1])  # Weight the distance by the property weight
        return sum(values) / len(values) if values else 0.0

    @classmethod
    def evaluate(cls, a: TDWorld, b: TDWorld, e: Entity) -> bool:
        cls.ensure_basics([a, b])
        values = []
        for k, v in e.properties.items():
            if k not in a or k not in b:
                raise KeyError()
            d = v[0].evaluate(a[k], b[k])
            values.append(d * v[1])  # Weight the distance by the property weight
        return all(values) if values else False

        # return cls.distance(a, b, e) == 0.0
