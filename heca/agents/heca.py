from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from itertools import permutations
from typing import Sequence
import torch
from torch.distributions import Categorical
from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.data import HeteroData, Batch

from heca.conditions.analyzer import ConditionAnalyzer
from heca.conditions.condition import Condition
from heca.conditions.pair import ConditionPair
from heca.agents.agent import Agent, AgentFeedback
from heca.conditions.evaluator import Evaluator
from heca.heca_gnn.network import Network
from heca.misc.ppo import PPO
from heca.misc.td import TDScene


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        evaluator: Evaluator.Config
        network: Network.Config
        agents: Sequence[Agent.Config]
        mode: HecaMode
        ppo: PPO.Config
        label: str = "heca"
        n_samples: int = 1000
        visualize: bool = True
        merge_threshold: float = 0.75

    def __init__(self, cfg: Config):
        self.cfg = cfg
        if self.cfg.mode == HecaMode.TRAIN:
            self.ppo = PPO.get(self.cfg.ppo).setup(self.cfg.network)
            self.network = self.ppo.copy_network()
        else:
            self.network = Network.get(cfg.network).eval()

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

        self.evaluator = Evaluator.get(cfg.evaluator).setup(self.conditions)
        self.analyzer = ConditionAnalyzer()

    def predict(
        self,
        data: HeteroData,
        options: list[tuple[Agent.Config, TDScene, TDScene]],
    ) -> tuple[Agent.Config, TDScene, TDScene]:
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

    def step(self, x: TDScene) -> tuple[TDScene, AgentFeedback]:
        data, options = self.generate_graph(x)
        sa, sx, sy = self.predict(data, options)
        z = Agent.get(sa).act(sx, sy)
        fb = self.evaluator.step(z)
        if self.cfg.mode == HecaMode.TRAIN:
            if self.ppo.post_action(fb):  # true if batch full
                raise NotImplementedError()
        return z, fb

    def act(self, x: TDScene, y: TDScene) -> TDScene:
        self.evaluator.reset(y)
        self.condition_graph(y)
        z, fb = self.step(x)
        while not fb.terminal:
            z, fb = self.step(z)
        return z

    def generate_graph(self, x: TDScene) -> tuple[
        HeteroData,
        list[tuple[Agent.Config, TDScene, TDScene]],
    ]:
        raise NotImplementedError()

    def condition_graph(self, y: TDScene):
        raise NotImplementedError()

    def precompute_graph(self, agents: list[Agent.Config]):
        cons: dict[Agent.Config, list[tuple[Condition, Condition]]] = {}
        for a, b in permutations(agents, 2):
            tuples = [(cfg, con) for con in Agent.get(cfg).conditions]
            cons.extend(tuples)
        sets = [{i} for i in range(len(cons))]
        while True:
            merged = False
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    a = cons[i]
                    b = cons[j]
                    sim_rating = self.analyzer.compute_sim(a, b)

    def to_batch(self, data: list[HeteroData]) -> Batch:
        return cast(Batch, Batch.from_data_list(data))  # type: ignore

    def sample(self) -> tuple[TDScene, TDScene]:
        raise NotImplementedError

    @cached_property
    def conditions(self) -> list[ConditionPair]:
        path = Agent.instance_dir(self.cfg)
        cons = []
        for cfg in self.cfg.agents:
            cons.extend(Agent.get(cfg).conditions)

        sets = [{i} for i in range(len(cons))]
        while True:
            merged = False
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    a = cons[i]
                    b = cons[j]
                    sim_rating = self.analyzer.compute_sim(a, b)
                    self.analyzer.plot_similarity(sim_rating, a, b, path)
                    if self.analyzer.evaluate_merge(
                        sim_rating, self.cfg.merge_threshold
                    ):
                        a_set = sets[i]
                        b_set = sets[j]
                        new_set = a_set | b_set
                        ids = map(str, sorted(new_set))
                        label = f"cond_pair_{''.join(ids)}"
                        new_pair = ConditionPair.merge(
                            label=label, a=a, b=b, n_samples=self.cfg.n_samples
                        )
                        new_pair.plot(path)
                        cons.pop(j)
                        cons.pop(i)
                        sets.pop(j)
                        sets.pop(i)
                        sets.append(new_set)
                        cons.append(new_pair)

                        merged = True
                        break
                if merged:
                    break
            if not merged:
                break
        return cons
