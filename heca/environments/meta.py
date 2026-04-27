from heca.entities.entity import Entity
from heca.environments.environment import Environment
from heca.misc.td import TDEntities, TDScene, TDWorld


class MetaEnvironment:
    @classmethod
    def sample(
        cls, envs: list[Environment.Query], e: Entity
    ) -> tuple[TDWorld, TDWorld]:
        x_s = dict()
        y_s = dict()
        for query in envs:
            env = Environment.search(query)
            x_v = env.sample()
            y_v = env.sample()
            x_s[query.label] = TDScene(heca=TDEntities(x_v))
            y_s[query.label] = TDScene(heca=TDEntities(y_v))
        return TDWorld(scenes=x_s), TDWorld(scenes=y_s)
