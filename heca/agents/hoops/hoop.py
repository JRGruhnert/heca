from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Type

import torch
from torch.distributions import Categorical
from heca.agents.leafs.leaf import LeafAgent
from heca.entities.entity import Entity
from heca.environments.environment import Environment
from heca.evaluators.agent_hoop import HoopEvaluator
from heca.agents.agent import Agent, AgentFeedback
from heca.generators.generator import HoopGenerator
from heca.misc.classes import V
from heca.networks.heca.hoop import HoopNetwork

from heca.misc import logger
from heca.misc.explainer import HoopgnExplainer
from heca.misc.ppo import PPO
from heca.misc.td import TDEntities, TDScene
from torch_geometric.explain import CaptumExplainer, HeteroExplanation


class HoopAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        hoop: HoopNetwork.Config
        generator: HoopGenerator.Config
        evaluator: HoopEvaluator.Config
        ppo: PPO.Config
        agents: set[Agent.Query]
        training: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.hoop = HoopNetwork.from_config(cfg.hoop)
        self.generator = HoopGenerator.from_config(cfg.generator)
        self.evaluator = HoopEvaluator.from_config(cfg.evaluator)
        self.explainer = HoopgnExplainer(
            self.hoop,
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
        self.ppo = PPO(cfg.ppo, hoop=self.hoop)

    def act(self, x: TDScene, y: TDScene, e: Entity) -> tuple[TDScene, AgentFeedback]:
        z = self.apply_expert_knowledge(y, e)
        if self.cfg.training:
            with torch.no_grad():
                variants, logits, value = self.predict(x, z)
                dist = Categorical(logits=logits)
                action = dist.sample()
        else:
            variants, logits, value = self.predict(x, z)
            action = logits.argmax(dim=-1)

        agent, entity = variants[action]
        z, fb = Agent.search(agent).act(x, z, entity)
        reward, done, terminal = self.evaluator.step(z, fb)
        if self.cfg.training:
            logprob: torch.Tensor = dist.log_prob(action)
            can_learn = self.ppo.step(
                x,
                y,
                action.detach(),
                logprob.detach(),
                value.detach(),
                reward,
                done,
                terminal,
            )
        else:
            can_learn = False
        return z, AgentFeedback(
            reward=reward, done=done, terminal=terminal, can_learn=can_learn
        )

    def predict(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, Entity]], torch.Tensor, torch.Tensor]:
        x, y = self.hoop.encode(x, y)
        options, data = self.generator(x, y)
        logits = self.hoop.actor(data)
        value = self.hoop.critic(data)
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

    def train(self, episodes: int, e: Entity):
        for ep in range(episodes):
            x, y = self.sample(e)
            terminal = False
            while not terminal:
                reward, fb = self.act(x, y, e)
                if fb.can_learn:
                    xp, highscore = self.ppo.learn(ep)
                    self.hoop.upgrade(xp)
                    self.hoop.save(
                        self.hoop.cfg.query,
                        epoch=ep,
                        label="best" if highscore else "latest",
                    )
                terminal = fb.terminal
                logger.info(f"Epoch {ep}: Reward={reward:.4f}, Done={terminal}")

    def explain(self, e: Entity):
        x, y = self.sample(e)
        x, y = self.hoop.encode(x, y)
        _, data = self.generator(x, y)
        action = self.hoop.actor(data)  # Forward pass to populate
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
