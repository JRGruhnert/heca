from heca.agents.agent import Agent
from heca.environments.scene import Scene
from heca.misc.td import TDWorld


class MetaWorld:
    register: set[Scene.Query] = set()

    @classmethod
    def search_and_register(cls, queries: list[Agent.Query]):
        for aq in queries:
            agent = Agent.search(aq)
            for sq in agent.required_scenes():
                cls.register.add(sq)

    @classmethod
    def sample(cls) -> TDWorld:
        scenes = dict()
        for query in cls.register:
            scenes[query.label] = Scene.search(query).sample()
        return TDWorld(scenes=scenes)

    @classmethod
    def get(cls, query: Scene.Query) -> Scene:
        if query.label in cls.register:
            return Scene.search(query)
        else:
            raise ValueError(f"Scene {query.label} not found in MetaWorld.")
