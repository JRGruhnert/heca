from dataclasses import dataclass

import numpy as np
from tensordict import TensorDict

from hoopgn import logger
from hoopgn.agents.agent import Agent
from hoopgn.environments.environment import Environment

from hoopgn.observation.converters.converter import Converter
from hoopgn.observation.td_scene import TDScene
from hoopgn.policies.policy import Policy


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Signature(Agent.Signature):
        label: str
        environment: Environment.Signature

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        sig: "LeafAgent.Signature"
        policy: Policy.Config

        v1: Converter.Config
        v2: Converter.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = Policy.from_config(self.cfg.policy)
        self.v1 = Converter.from_config(self.cfg.v1)
        self.v2 = Converter.from_config(self.cfg.v2)

    @classmethod
    def count_by_environment(cls, label: str | None = None) -> int:
        if label is None:
            return len(cls.registry)
        return sum(1 for c in cls.registry if c.signature.environment.label == label)

    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        env = Environment.get(self.cfg.sig.environment)
        # self.modify()
        self.policy.reset(y)

        while (action := self.policy(x.leaf)) is not None:
            feedback = env.step(action)
            c = env.observation()
            x = self.convert(c)
        reward, done = self.evaluator.step(self.current, self.goal)
        self.current_step += 1
        terminal = True if self.current_step >= self.max_allowed_steps else done
        logger.info(
            f"Step {self.current_step}: Reward={reward}, Done={done}, Terminal={terminal}"
        )
        return self.current, reward, done, terminal
        raise NotImplementedError()
        # return self.policy(x, y)

    def predict(self, x: TDScene) -> np.ndarray | None:
        return self.policy(x)

    def convert(self, obs) -> TDScene:
        return {
            "v1": self.v1(obs),
            "v2": self.v2(obs),
        }

    def convert_leaf(self, obs) -> TensorDict:
        return {
            "v1": self.v1(obs),
            "v2": self.v2(obs),
        }
