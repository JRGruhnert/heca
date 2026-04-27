from dataclasses import dataclass, field
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Type

import torch
from torch.distributions import Categorical
from heca.agents.leafs.leaf import LeafAgent
from heca.entities.entity import Entity
from heca.entities.meta import MetaEntity
from heca.environments.environment import Environment
from heca.environments.meta import MetaEnvironment
from heca.evaluators.heca import HecaEvaluator
from heca.agents.agent import Agent, AgentFeedback
from heca.generators.generator import HecaGenerator
from heca.heca_gnn.network import HecaNetwork
from heca.misc.ppo import PPO
from heca.misc.td import TDEntities, TDScene
from torch_geometric.explain import CaptumExplainer, Explainer

from typing import TypeVar, Type

V = TypeVar("V", bound="Heca")


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        generator: HecaGenerator.Config
        evaluator: HecaEvaluator.Config
        network: HecaNetwork.Query
        agents: set[Agent.Query]
        ppo: PPO.Query
        mode: HecaMode = HecaMode.EVAL

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.mode == HecaMode.TRAIN:
            self.ppo = PPO.load(self.cfg.ppo, self.cfg.network)
            self.network = self.ppo.collector()
        else:
            self.network = HecaNetwork.load(cfg.network)

        if self.cfg.mode == HecaMode.EXPLAIN:
            self.explainer = Explainer(
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

        self.generator = HecaGenerator.create(cfg.generator)
        self.evaluator = HecaEvaluator.create(cfg.evaluator)
        # TODO: Create meta here correctly and post vs precons vs both???
        agents = Agent.search_multiple(list(self.cfg.agents))
        precons: list[Entity] = []
        for agent in agents:
            precons.extend(agent.precons)
        self.meta = MetaEntity.merge(precons, self.cfg.network.label)

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        z = self.apply_expert_knowledge(y, self.meta)
        if self.cfg.mode == HecaMode.EXPLAIN:
            variants, logits, value = self.predict(x, z)
        else:
            with torch.no_grad():
                variants, logits, value = self.predict(x, z)

        if self.cfg.mode == HecaMode.TRAIN:
            dist = Categorical(logits=logits)
            action = dist.sample()
        else:
            action = logits.argmax(dim=-1)

        agent, entity = variants[action]
        z, fb = Agent.search(agent).act(x, z, entity)
        reward, done, terminal = self.evaluator.step(z, fb)
        if self.cfg.mode == HecaMode.TRAIN:
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
        return z, AgentFeedback(
            reward=reward,
            done=done,
            terminal=terminal,
            can_learn=can_learn,
        )

    def predict(
        self, x: TDScene, y: TDScene
    ) -> tuple[list[tuple[Agent.Query, Entity]], torch.Tensor, torch.Tensor]:
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

    def sample(self, e: Entity) -> tuple[TDScene, TDScene]:
        envs: list[Environment.Query] = []
        x, y = MetaEnvironment.sample(envs, e)
        attempts = 0
        while not self.evaluator.is_sample(x, y, e):
            attempts += 1
            if attempts % 5 == 0:
                x_values = dict()
                for env in self.environments:
                    x_values[env.cfg.query.label] = env.sample()
                x = TDScene(heca=TDEntities(x_values))
            for env in self.environments:
                y_values[env.cfg.query.label] = env.sample()
            y = TDScene(heca=TDEntities(y_values))
        return x, y

    @cached_property
    def precons(self) -> list[Entity]:
        raise NotImplementedError()

    @cached_property
    def postcons(self) -> list[Entity]:
        raise NotImplementedError()

    @classmethod
    def resolve_path(cls: Type[V], query: "Heca.Query") -> Path:
        raise NotImplementedError()

    def load(self, path: Path, label: str) -> None:
        raise NotImplementedError()

    def save(self, path: Path, label: str) -> None:
        raise NotImplementedError()
