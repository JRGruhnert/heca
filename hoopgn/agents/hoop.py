from dataclasses import dataclass, field
from functools import cached_property

import torch
from torch.distributions import Categorical
from hoopgn.agents.leafs.leaf import LeafAgent
from hoopgn.environments.environment import Environment
from hoopgn.evaluators.evaluator import Evaluator
from hoopgn.agents.agent import Agent, AgentFeedback
from hoopgn.generators.generator import Generator
from hoopgn.networks.hoops.hoop import HoopNetwork

from hoopgn.misc import logger
from hoopgn.misc.explainer import HoopgnExplainer
from hoopgn.misc.ppo import PPO
from hoopgn.misc.td import TDScene, TDEntity
from torch_geometric.explain import CaptumExplainer, HeteroExplanation


class HoopAgent(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        hoop: HoopNetwork.Config
        generator: Generator.Config
        evaluator: Evaluator.Config
        reinforcement: PPO.Config
        agents: set[Agent.Query]
        environments: set[Environment.Query] = field(init=False)
        training: bool = False
        max_steps: int = 10

        def __post_init__(self):
            environments = set()
            for query in self.agents:
                agent = Agent.search(query)
                if isinstance(agent, LeafAgent):
                    environments.add(agent.cfg.query.parent)
                elif isinstance(agent, HoopAgent):
                    environments.update(agent.cfg.environments)
                else:
                    raise ValueError()
            self.environments = environments

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.network = HoopNetwork.from_config(cfg.hoop)
        self.generator = Generator.from_config(cfg.generator)
        self.evaluator = Evaluator.from_config(cfg.evaluator)
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

    def act(self, x: TDScene, y: TDScene) -> tuple[TDScene, AgentFeedback]:
        for step in range(self.cfg.max_steps):
            if self.cfg.training:
                with torch.no_grad():
                    variants, logits, value = self.predict(x, y)
                    dist = Categorical(logits=logits)
                    action = dist.sample()
            else:
                variants, logits, value = self.predict(x, y)
                action = logits.argmax(dim=-1)

            s_a, s_x, s_y = variants[action]
            z, feedback = Agent.search(s_a).act(s_x, s_y)
            reward, done = self.evaluator.step(z, feedback)
            if step == self.cfg.max_steps - 1:
                terminal = True
            else:
                terminal = done

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

                if terminal:
                    # Stop early if done
                    break

        return z, self.evaluator.get_feedback(z)

    def predict(self, x: TDScene, y: TDScene) -> tuple[
        list[tuple[Agent.Query, TDScene, TDScene]],
        torch.Tensor,
        torch.Tensor,
    ]:
        x, y = self.network.encode(x, y)
        options, data = self.generator(x, y)
        logits = self.network.actor(data)
        value = self.network.critic(data)
        return options, logits, value

    def sample(self) -> tuple[TDScene, TDScene]:
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
        while not self.evaluator.is_sample(x, y):
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
            x, y = self.sample()
            done = False
            while not done:
                reward, done = self.act(x, y)
                logger.info(f"Epoch {epoch}: Reward={reward:.4f}, Done={done}")

    def explain(self):
        x, y = self.sample()
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
    def precons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()

    @cached_property
    def postcons(self) -> dict[str, TDEntity]:
        raise NotImplementedError()
