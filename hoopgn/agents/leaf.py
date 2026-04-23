import torch

from dataclasses import dataclass

from tensordict import TensorDict

from hoopgn import observation
from hoopgn.agents.agent import Agent
from hoopgn.entities.entity import Entity
from hoopgn.misc import logger
from hoopgn.policies.leafs.leaf import LeafPolicy
from hoopgn.properties.property import Property
from hoopgn.environments.environment import Environment
from hoopgn.observation.converters.calvin_td import LeafConverter
from hoopgn.observation.converters.converter import Converter
from hoopgn.observation.td_entity import TDEntity
from hoopgn.observation.td_scene import TDScene


class LeafAgent(Agent):
    @dataclass(kw_only=True)
    class Signature(Agent.Signature):
        label: str
        environment: Environment.Signature

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        sig: "LeafAgent.Signature"
        policy: LeafPolicy.Config

        leaf_cv: LeafConverter.Config
        env_cv: Converter.Config

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = LeafPolicy.from_config(self.cfg.policy)
        self.leaf_cv = LeafConverter.from_config(self.cfg.leaf_cv)
        self.env_cv = Converter.from_config(self.cfg.env_cv)

        # Loading #TODO: correct path
        self.policy.from_disk(f"data/policies/{self.cfg.sig.label}.json")

    def get_agent_td(self, x: TDScene) -> TDScene:
        return x.policies.get(self.cfg.sig.environment.label)

    def make_td_scene(self, ax: TensorDict) -> TDScene:
        v1, v2 = self.leaf_cv(self.current)
        return TDScene(
            v1=v1,
            v2=v2,
            policies=TensorDict(
                {self.cfg.sig.environment.label: ax},
                observation.empty_batchsize,
            ),
        )

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

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, float, bool]:
        env = Environment.get(self.cfg.sig.environment)
        ax = self.get_agent_td(x)
        ay = self.get_agent_td(y)

        # Policy loop
        while (action := self.policy(ax, ay)) is not None:
            self.current = env.step(action)
            ax = self.env_cv(self.current)

        scene = self.make_td_scene(ax)
        reward, done = self.evaluator.step(scene)
        logger.debug(f"Step {self.step_counter}: Reward={reward}, Done={done}")
        return scene, reward, done

    @property
    def ppre(self) -> dict[Property.Signature, torch.Tensor]:
        # TODO: Cluster from previous layer
        return self.policy.ppre

    @property
    def ppost(self) -> dict[Property.Signature, torch.Tensor]:
        # TODO: Cluster from previous layer
        return self.policy.ppost

    @property
    def epre(self) -> dict[Entity.Signature, TDEntity]:
        # TODO: Cluster from previous layer
        return self.policy.epre

    @property
    def epost(self) -> dict[Entity.Signature, TDEntity]:
        # TODO: Cluster from previous layer
        return self.policy.epost
