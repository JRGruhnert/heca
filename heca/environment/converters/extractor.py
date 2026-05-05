from dataclasses import dataclass

from heca.classes.persist import Persistable
from heca.misc.td import TDScene

# NOTE: copied and adapted from TAPAS (https://github.com/robot-learning-freiburg/TAPAS.git)


class Extractor(Persistable):
    @dataclass(frozen=True, kw_only=True)
    class Query(Persistable.Query):
        pass

    @dataclass(frozen=True, kw_only=True)
    class File(Persistable.File):
        pass

    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        pass

    def __call__(self, obs) -> tuple[TDScene, bool]:
        raise NotImplementedError()
