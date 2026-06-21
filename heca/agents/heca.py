from dataclasses import dataclass
from enum import Enum
from functools import cached_property

import torch
from torch.distributions import Categorical
from heca.misc.entity import Entity
from heca.evaluators.heca import HecaEvaluator
from heca.agents.agent import Agent, AgentFeedback
from heca.generators.generator import HecaGenerator
from heca.heca_gnn.network import HecaNetwork
from heca.misc.ppo import PPO
from heca.misc.td import TDScene
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

    def predict(self, data, options) -> tuple[Agent.Config, TDScene, TDScene]:
        with torch.inference_mode():
            logits = self.network.actor(data)

        if self.cfg.mode == HecaMode.TRAIN:
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob: torch.Tensor = dist.log_prob(action)
            with torch.inference_mode():
                value = self.network.critic(data)
            self.ppo.pre_action(data, action, logprob, value)
        else:
            action = logits.argmax(dim=-1)

        return options[action]

    def step(self, data, options) -> tuple[TDScene, AgentFeedback]:
        agent, x, y = self.predict(data, options)
        z = Agent.get(agent).act(x, y)
        fb = self.evaluator.step(z)
        if self.cfg.mode == HecaMode.TRAIN:
            learn = self.ppo.post_action(
                fb.reward,
                fb.done,
                fb.terminal,
            )
            if learn:
                raise NotImplementedError()
        return z, fb

    def act(self, x: TDScene, y: TDScene) -> TDScene:
        self.evaluator.reset(x, y)
        data, options = self.generator.reset(x, y)
        z, fb = self.step(data, options)
        while not fb.terminal:
            data, options = self.generator.step(z)
            z, fb = self.step(data, options)
        return z

    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError
        # x = MetaWorld.sample()
        # y = MetaWorld.sample()
        # attempts = 0
        # while not self.evaluator.is_sample(x, y):
        #     attempts += 1
        #     if attempts % 5 == 0:
        #         x = MetaWorld.sample()
        #     y = MetaWorld.sample()
        # return x, y

    @cached_property
    def precons(self) -> list[Entity]:
        raise NotImplementedError

    @cached_property
    def postcons(self) -> list[Entity]:
        raise NotImplementedError
