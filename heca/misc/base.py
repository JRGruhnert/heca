import abc
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Type, TypeVar, cast

from heca.misc import logger

C = TypeVar("C", bound="Configurable")
R = TypeVar("R", bound="Registerable")
P = TypeVar("P", bound="Persistable")


class Configurable(abc.ABC):
    _config_registry: ClassVar[dict[type, type]] = {}

    @dataclass(kw_only=True)
    class Config:
        pass

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._config_registry = {}
        cls._instances = {}
        # Register cls.Config -> cls into every Registerable ancestor's registry,
        # so A.get(B.Config(...)) resolves B even when called on A.
        own_config = cls.__dict__.get("Config")
        if own_config is None:
            # raise TypeError(f"{cls.__name__} must define a nested Config dataclass.")
            return

        for base in cls.__mro__[1:]:
            if issubclass(base, Configurable):
                base._config_registry[own_config] = cls

    def __init__(self, cfg: "Configurable.Config"):
        self.cfg = cfg

    @classmethod
    def create(cls: Type[C], cfg: "Configurable.Config") -> C:
        target_cls = cls._config_registry.get(type(cfg), cls)
        return target_cls(cfg)


class Registerable(Configurable):
    _instances: ClassVar[dict[tuple[type, str], "Registerable"]] = {}

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str

    def __init__(self, cfg: "Registerable.Config"):
        super().__init__(cfg)

    @classmethod
    def create(cls, cfg):
        raise RuntimeError(
            f"{cls.__name__} is Registerable and must be instantiated via get()."
        )

    @classmethod
    def get(cls: Type[R], cfg: "Registerable.Config") -> R:
        # Resolve the concrete subclass from the config type; fall back to cls
        # itself if this is already a concrete class being called directly.
        target_cls = cls._config_registry.get(type(cfg), cls)
        key = (type(cfg), cfg.label)
        if key not in cls._instances:
            cls._instances[key] = target_cls(cfg)
        return cast(R, cls._instances[key])


class Persistable(Registerable, abc.ABC):  # (ABC, metaclass=ConfigurableMeta):
    root: ClassVar[Path] = Path("data")
    _persisted_instances: ClassVar[dict[tuple[str, str], "Persistable"]] = {}

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        cls._persisted_instances = {}

    @dataclass(kw_only=True)
    class Config(Registerable.Config):
        subroot: str
        folder: str

    def __init__(self, cfg: "Persistable.Config"):
        super().__init__(cfg)

    @classmethod
    def create(cls, cfg):
        raise RuntimeError(
            f"{cls.__name__} is Persistable and must be instantiated via load()."
        )

    @classmethod
    def get(cls, cfg):
        raise RuntimeError(
            f"{cls.__name__} is Persistable and must be accessed via load()."
        )

    @classmethod
    def load(cls: Type[P], cfg: "Persistable.Config", skip_loading: bool = False) -> P:
        target_cls = cls._config_registry.get(type(cfg), cls)
        key = (cfg.folder, cfg.label)
        if key not in cls._persisted_instances:
            instance = target_cls(cfg)
            assert isinstance(instance, Persistable)
            path = cls.resolve(cfg)
            if not skip_loading:
                logger.info(f"Loading {type(instance)} {cfg.label} data from {path}")
                instance._load(path)
            cls._persisted_instances[key] = instance
        return cast(P, cls._persisted_instances[key])

    @classmethod
    def save(cls: Type[P], cfg: "Persistable.Config"):
        key = (cfg.folder, cfg.label)
        if key not in cls._persisted_instances:
            raise KeyError(
                f"No instance found for folder={cfg.folder!r}, label={cfg.label!r}"
            )
        persistable = cast(P, cls._persisted_instances[key])
        path = cls.resolve(cfg)
        logger.info(f"Saving {type(persistable)} {cfg.label} data to {path}")
        persistable._save(path)

    @classmethod
    def resolve_base(cls, cfg: "Persistable.Config") -> Path:
        path = cls.root / cfg.subroot / cfg.label
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def resolve(cls, cfg: "Persistable.Config") -> Path:
        path = cls.resolve_base(cfg) / cfg.folder
        path.mkdir(parents=True, exist_ok=True)
        return path

    @abc.abstractmethod
    def _load(self, path: Path):
        raise NotImplementedError()

    @abc.abstractmethod
    def _save(self, path: Path) -> bool:
        raise NotImplementedError()
