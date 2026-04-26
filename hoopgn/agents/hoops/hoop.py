from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Type

import torch
from torch.distributions import Categorical
from hoopgn.agents.leafs.leaf import LeafAgent
from hoopgn.entities.entity import Entity
from hoopgn.environments.environment import Environment
from hoopgn.evaluators.agent_hoop import HoopEvaluator
from hoopgn.agents.agent import Agent, AgentFeedback
from hoopgn.generators.generator import Generator
from hoopgn.misc.classes import V
from hoopgn.networks.hoops.hoop import HoopNetwork

from hoopgn.misc import logger
from hoopgn.misc.explainer import HoopgnExplainer
from hoopgn.misc.ppo import PPO
from hoopgn.misc.td import TDEntities, TDScene
from torch_geometric.explain import CaptumExplainer, HeteroExplanation


class HoopAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        hoop: HoopNetwork.Config
        generator: Generator.Config
        evaluator: HoopEvaluator.Config
        reinforcement: PPO.Config
        agents: set[Agent.Query]
        training: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.network = HoopNetwork.from_config(cfg.hoop)
        self.generator = Generator.from_config(cfg.generator)
        self.evaluator = HoopEvaluator.from_config(cfg.evaluator)
        self.reinforcement = PPO.from_config(cfg.reinforcement)
        self.explainer = HoopgnExplainer(
            self.network,
            algorithm=CaptumExplainer("Saliency"),
            explanation_type="model",
            node_mask_type="attributes",
            edge_mask_type="object",
            model_config=dict(
                mode="multiclass_classification",
                task_level="node",
                return_type="probs",
            ),
        )

    def act(
        self, x: TDScene, y: TDScene, e: Entity | None = None
    ) -> tuple[TDScene, AgentFeedback]:
        z = self.apply_expert_knowledge(y, e)
        if self.cfg.training:
            with torch.no_grad():
                variants, logits, value = self.predict(x, z)
                dist = Categorical(logits=logits)
                action = dist.sample()
        else:
            variants, logits, value = self.predict(x, z)
            action = logits.argmax(dim=-1)

        a, e = variants[action]
        z, fb = Agent.search(a).act(x, z, e=e)
        reward, done, terminal = self.evaluator.step(z, fb)
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

        return z, AgentFeedback(reward=reward, done=done, terminal=terminal)

    def predict(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, Entity | None]], torch.Tensor, torch.Tensor]:
        x, y = self.network.encode(x, y)
        options, data = self.generator(x, y)
        logits = self.network.actor(data)
        value = self.network.critic(data)
        return options, logits, value

    def cluster_precon(self) -> dict[str, Entity]:
        raise NotImplementedError()

    def cluster_postcon(self) -> dict[str, Entity]:
        raise NotImplementedError()

    def apply_expert_knowledge(self, x: TDScene, entity: Entity | None) -> TDScene:
        raise NotImplementedError()
        if entity is None:
            return x
        else:
            self.postcons

    @property
    def environments(self) -> list[Environment]:
        envs = set()
        for agent_query in self.cfg.agents:
            agent = Agent.search(agent_query)
            if isinstance(agent, LeafAgent):
                envs.add(agent.cfg.query.env)
            elif isinstance(agent, HoopAgent):
                envs.update(agent.environments)
        return [Environment.search(cfg) for cfg in envs]

    def sample(self, e: Entity) -> tuple[TDScene, TDScene]:
        x_values = dict()
        y_values = dict()
        for env in self.environments:
            x_values[env.cfg.query.label] = env.sample()
            y_values[env.cfg.query.label] = env.sample()

        x = TDScene(data=TDEntities(x_values))
        y = TDScene(data=TDEntities(y_values))
        attempts = 0
        while not self.evaluator.is_sample(x, y, e):
            attempts += 1
            if attempts % 5 == 0:
                x_values = dict()
                for env in self.environments:
                    x_values[env.cfg.query.label] = env.sample()
                x = TDScene(data=TDEntities(x_values))
            for env in self.environments:
                y_values[env.cfg.query.label] = env.sample()
            y = TDScene(data=TDEntities(y_values))
        return x, y

    def train(self, epochs: int, e: Entity):
        for epoch in range(epochs):
            x, y = self.sample(e)
            terminal = False
            while not terminal:
                reward, fb = self.act(x, y, e=e)
                terminal = fb.terminal
                logger.info(f"Epoch {epoch}: Reward={reward:.4f}, Done={terminal}")

    def explain(self, e: Entity):
        x, y = self.sample(e)
        x, y = self.network.encode(x, y)
        _, data = self.generator(x, y)
        action = self.network.actor(data)  # Forward pass to populate
        explanation = self.explainer(
            data.x_dict,
            data.edge_index_dict,
            data.edge_attr_dict,
            index=action.argmax(dim=-1),
        )

        assert isinstance(explanation, HeteroExplanation)
        return explanation

    @cached_property
    def precons(self) -> list[Entity]:
        raise NotImplementedError()

    @cached_property
    def postcons(self) -> list[Entity]:
        raise NotImplementedError()

    @classmethod
    def resolve_path(cls: Type[V], query: "HoopAgent.Query") -> Path:
        raise NotImplementedError()

    def load(self, path: Path, label: str) -> None:
        raise NotImplementedError()

    def save(self, path: Path, label: str) -> None:
        raise NotImplementedError()
