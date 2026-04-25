from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch.distributions import Categorical
from hoopgn.agents.leafs.leaf import LeafAgent
from hoopgn.environments.environment import Environment
from hoopgn.misc import logger
from hoopgn.misc.ppo import PPO
from hoopgn.misc.td import TDScene, TDEntity

from hoopgn.hoops.hoop import Hoop
from hoopgn.agents.agent import Agent
from hoopgn.generators.generator import Generator


class BranchAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        generator: Generator.Config
        network: Hoop.Config
        agents: list[Agent.Query]
        reinforcement: PPO.Config
        training: bool = False
        environments: set[Environment.Query] = field(init=False)

        def __post_init__(self):
            temp = set()
            for query in self.agents:
                agent = Agent.search(query)
                if isinstance(agent, LeafAgent):
                    temp.add(agent.cfg.environment)
                elif isinstance(agent, BranchAgent):
                    temp.update(agent.cfg.environments)
                else:
                    raise ValueError(
                        f"Agent '{agent}' is neither a LeafAgent nor a BranchAgent."
                    )
            self.environments = temp

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.generator = Generator.from_config(cfg.generator)
        self.network = Hoop.from_config(cfg.network)
        self.reinforcement = PPO.from_config(cfg.reinforcement)

    def execute_variant(
        self, variant: tuple[Agent.Query, TDScene, TDScene]
    ) -> tuple[float, bool, bool]:
        agent = Agent.search(variant[0])
        reward, done, terminal = agent.act(variant[1], variant[2])
        return reward, done, terminal

    def act(self, x: TDScene, y: TDScene) -> tuple[float, bool, bool]:
        if self.cfg.training:
            with torch.no_grad():
                variants, logits, value = self.predict(x, y)
        else:
            variants, logits, value = self.predict(x, y)

        if self.cfg.training:
            dist = Categorical(logits=logits)
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        reward, done, terminal = self.execute_variant(variants[action])

        if self.cfg.training:
            logprob: torch.Tensor = dist.log_prob(action)
            full = self.reinforcement.append(
                x,
                y,
                action.detach(),
                logprob.detach(),
                value.detach(),
                reward,
                done,
                terminal,
            )
            if full:
                state_dict = self.reinforcement.learn()
                self.network.load_state_dict(state_dict)

        return reward, done, terminal

    def predict(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, TDScene, TDScene]], torch.Tensor, torch.Tensor]:
        x, y = self.network.preprocess(x, y)
        variants = self.generator(x, y)
        logits, value = self.network.forward(variants)
        return variants, logits, value

    def make_variants(
        self, x: TDScene, y: TDScene
    ) -> list[tuple[Agent.Query, TDScene, TDScene]]:
        raise NotImplementedError()

    def sample_task(self) -> tuple[TDScene, TDScene]:
        x_values = dict()
        y_values = dict()
        for cfg in self.cfg.environments:
            env = Environment.search(cfg)
            x_v = env.sample()
            y_v = env.sample()
            x_values[cfg.label] = x_v
            y_values[cfg.label] = y_v

        x = TDScene(x_values)
        y = TDScene(y_values)
        attempts = 0
        while not self.evaluator.check_sample(x, y):
            attempts += 1
            if attempts % 5 == 0:
                x_values = dict()
                for cfg in self.cfg.environments:
                    env = Environment.search(cfg)
                    x_values[cfg.label] = env.sample()
                x = TDScene(x_values)
            for cfg in self.cfg.environments:
                env = Environment.search(cfg)
                y_values[cfg.label] = env.sample()
            y = TDScene(y_values)
        return x, y

    def train(self, epochs: int):
        for epoch in range(epochs):
            x, y = self.sample_task()
            done = False
            while not done:
                reward, done, terminal = self.act(x, y)
                logger.info(
                    f"Epoch {epoch}: Reward={reward:.4f}, Done={done}, Terminal={terminal}"
                )

    @cached_property
    def precons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()

    @cached_property
    def postcons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
