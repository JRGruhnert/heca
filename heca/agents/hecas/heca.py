from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import torch
from torch.distributions import Categorical
from heca.entities.entity import Entity
from heca.environment.world import MetaWorld
from heca.evaluators.heca import HecaEvaluator
from heca.agents.agent import Agent, AgentFeedback
from heca.generators.generator import HecaGenerator
from heca.heca_gnn.network import HecaNetwork
from heca.misc.ppo import PPO
from heca.misc.td import TDScene, TDWorld
from torch_geometric.explain import CaptumExplainer, Explainer


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        generator: HecaGenerator.Config
        evaluator: HecaEvaluator.Config
        network: HecaNetwork.Config
        agents: set[Agent.Config]
        mode: HecaMode
        ppo: PPO.Config

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.mode == HecaMode.TRAIN:
            self.ppo = PPO.get(self.cfg.ppo)
            self.ppo.load_network(self.cfg.network)
            self.network = self.ppo.collector()
        else:
            self.network = HecaNetwork.get(cfg.network)

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

        self.generator = HecaGenerator.get(cfg.generator)
        self.evaluator = HecaEvaluator.get(cfg.evaluator)

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        # z = self.apply_expert_knowledge(y)
        z = y
        if self.cfg.mode == HecaMode.EXPLAIN:
            agent, xl, yl = self.predict(x, z)
        else:
            with torch.no_grad():
                agent, xl, yl = self.predict(x, z)

        az, afb = Agent.get(agent).act(xl, yl)
        reward, done, terminal = self.evaluator.step(az, afb)
        if self.cfg.mode == HecaMode.TRAIN:
            learn = self.ppo.step(
                xl,
                yl,
                action.detach(),
                logprob.detach(),
                value.detach(),
                reward,
                done,
                terminal,
            )
            if learn:
                raise NotImplementedError()
        return z, fb

    def predict(self, x: TDScene, y: TDScene) -> tuple[Agent.Config, TDScene, TDScene]:
        x, y = self.network.encode(x, y)
        options, data = self.generator(x, y)
        logits = self.network.actor(data)
        value = self.network.critic(data)

        if self.cfg.mode == HecaMode.TRAIN:
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob: torch.Tensor = dist.log_prob(action)
        else:
            action = logits.argmax(dim=-1)

        return options[action]

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
