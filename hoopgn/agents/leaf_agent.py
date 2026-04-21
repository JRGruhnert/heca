from dataclasses import dataclass

import numpy as np
from tensordict import TensorDict

from hoopgn import logger
from hoopgn.agents.agent import Agent
from hoopgn.environments.environment import Environment

from hoopgn.evaluators.evaluator import Evaluator
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

    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        env = Environment.get(self.cfg.sig.environment)
        self.evaluator.reset(y.leaf)
        # self.modify()
        self.policy.reset(y.leaf)

        while (action := self.policy(x.leaf)) is not None:
            z, reward, done = env.step(action)
            c = env.observation()
            self.policy.convert
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

    def sample_task(self) -> tuple[TDScene, TDScene]:
        env = Environment.get(self.cfg.sig.environment)
        logger.info("Sampling new Task...")
        self.step_counter = 0
        self.current = env.sample()
        self.goal = env.sample()
        attempts = 0
        while not self.evaluator.check_sample(self.current, self.goal):
            attempts += 1
            if attempts % 5 == 0:
                self.current = env.sample()
            self.goal = env.sample()
        return self.current, self.goal
