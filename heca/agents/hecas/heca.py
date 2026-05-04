from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from pathlib import Path
from typing import Type

import torch
from torch.distributions import Categorical
from heca.entities.entity import Entity
from heca.environment.scenes.scene import Scene
from heca.environment.world import MetaWorld
from heca.evaluators.heca import HecaEvaluator
from heca.agents.agent import Agent, AgentFeedback
from heca.generators.generator import HecaGenerator
from heca.heca_gnn.network import HecaNetwork
from heca.misc.ppo import PPO
from heca.misc.td import TDScene, TDWorld
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
        mode: HecaMode
        ppo: PPO.Query

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

        MetaWorld.search_and_register(list(self.cfg.agents))

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        # z = self.apply_expert_knowledge(y)
        z = y
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
        z, fb = Agent.search(agent).act(x, z)
        reward, done, terminal = self.evaluator.step(z, fb)
        if self.cfg.mode == HecaMode.TRAIN:
            logprob: torch.Tensor = dist.log_prob(action)
            learn = self.ppo.step(
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
            learn=learn,
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

    def sample(self, e: Entity) -> tuple[TDWorld, TDWorld]:
        x = MetaWorld.sample()
        y = MetaWorld.sample()
        attempts = 0
        while not self.evaluator.is_sample(x, y, e):
            attempts += 1
            if attempts % 5 == 0:
                x = MetaWorld.sample()
            y = MetaWorld.sample()
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

    def required_scenes(self) -> list[Scene.Query]:
        scenes = set()
        for agent_query in self.cfg.agents:
            agent = Agent.search(agent_query)
            for scene_query in agent.required_scenes():
                scenes.add(scene_query)
        return list(scenes)
