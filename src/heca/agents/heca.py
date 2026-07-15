from dataclasses import dataclass
from enum import Enum
from functools import cached_property
from typing import Sequence
import torch
from torch.distributions import Categorical
from torch_geometric.explain import Explainer, CaptumExplainer
from heca.conditions.pair import ConPair
from heca.agents.agent import Agent, AgentFeedback
from heca.conditions.evaluator import Evaluator
from heca.graphs.graph import Graph
from heca.heca_gnn.network import Network
from heca.misc import logger
from heca.misc.data import DCScene
from heca.misc.entity import Entity
from heca.misc.ppo import PPO
from heca.scenes.scene import Scene


class HecaMode(Enum):
    TRAIN = "train"
    EXPLAIN = "explain"
    EVAL = "evaluate"


class Heca(Agent):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        evaluator: Evaluator.Config = Evaluator.Config()
        network: Network.Config = Network.Config()
        agents: Sequence[Agent.Config]
        mode: HecaMode = HecaMode.TRAIN
        ppo: PPO.Config = PPO.Config()
        label: str = "heca"
        visualize: bool = True
        n_samples: int = 1000
        threshold: float = 0.75

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

        self.evaluator = Evaluator.get(cfg.evaluator).setup(
            self.conditions, self.entities
        )
        self.graph = Graph().generate(list(self.cfg.agents), self.entities)

    def predict(self) -> tuple[Agent.Config, DCScene]:
        data = self.graph.export()
        with torch.inference_mode():
            logits = self.network.actor(data)

        if self.cfg.mode == HecaMode.TRAIN:
            dist = Categorical(logits=logits)
            action = dist.sample()
            logprob: torch.Tensor = dist.log_prob(action)
            with torch.inference_mode():
                value = self.network.critic(data)
            self.ppo.store_action(data, action, logprob, value)
        else:
            action = logits.argmax(dim=-1)

        return self.graph.select(int(action))

    def step(self, x: DCScene) -> tuple[DCScene, AgentFeedback]:
        self.graph.set_start(x)
        a, y = self.predict()
        z = Agent.get(a).act(x, y)
        fb = self.evaluator.step(z)
        if self.cfg.mode == HecaMode.TRAIN:
            self.ppo.store_feedback(fb)
        return z, fb

    def act(self, x: DCScene, y: DCScene) -> DCScene:
        self.graph.set_goal(y)
        self.evaluator.reset(y)
        z, fb = self.step(x)
        while not fb.terminal:
            z, fb = self.step(z)
        return z

    def sample(self, cfg: Scene.Config) -> tuple[DCScene, DCScene]:
        scene = Scene.get(cfg)
        (x, ix), (y, iy) = scene.sample_task()
        while not self.evaluator.test_task(x, y):
            (x, ix), (y, iy) = scene.sample_task()
        return x, y

    def train(self, max_episodes: int, cfg: Scene.Config):
        """Train the network with PPO for a given number of episodes."""
        episodes_in_batch = 0

        for ep in range(max_episodes):
            x, y = self.sample(cfg)
            z = self.act(x, y)  # runs a full episode to terminal, accumulates PPO data
            episodes_in_batch += 1
            ep_reward = 0.0
            ep_len = 0
            for r, t in zip(reversed(self.ppo.rewards), reversed(self.ppo.terminals)):
                ep_reward += r
                ep_len += 1
                if t:  # reached the terminal step of this episode
                    break

            logger.info(f"Episode {ep}: len={ep_len}, reward={ep_reward:.4f}")

            # Perform PPO update once the rollout buffer is full (or overfull).
            if len(self.ppo.data) >= self.ppo.cfg.batch_size:
                progress = ep / max_episodes
                state_dict, is_best = self.ppo.learn(progress)

                # Sync the inference network with the learned policy
                self.network.load_state_dict(state_dict)

                # Log batch-level stats
                batch_success = sum(
                    1 for s in self.ppo.success[: self.ppo.cfg.batch_size] if s
                )
                success_rate = batch_success / self.ppo.cfg.batch_size
                logger.info(
                    f"PPO update: success_rate={success_rate:.2f}, "
                    f"episodes_in_batch={episodes_in_batch}, "
                    f"highscore={self.ppo.highscore:.4f}"
                )
                episodes_in_batch = 0

    @cached_property
    def elabels(self) -> set[str]:
        values = set()
        for cfg in self.cfg.agents:
            for con in Agent.get(cfg).conditions:
                values.union(con.elabels)
        return values

    @cached_property
    def entities(self) -> set[Entity]:
        values = set()
        for cfg in self.cfg.agents:
            values.union(Agent.get(cfg).entities)
        return values

    @cached_property
    def conditions(self) -> list[ConPair]:
        path = Agent.load_dir(self.cfg)
        cons: list[ConPair] = []
        for cfg in self.cfg.agents:
            cons.extend(Agent.get(cfg).conditions)

        sets = [{i} for i in range(len(cons))]
        while True:
            merged = False
            for i in range(len(cons)):
                for j in range(i + 1, len(cons)):
                    a = cons[i]
                    b = cons[j]
                    if a.can_merge(b, path):
                        a_set = sets[i]
                        b_set = sets[j]
                        new_set = a_set | b_set
                        ids = map(str, sorted(new_set))
                        label = f"{self.cfg.tag}_{''.join(ids)}"
                        new_pair = ConPair.merge(
                            label=label,
                            a=a,
                            b=b,
                            n_samples=self.cfg.n_samples,
                            threshold=self.cfg.threshold,
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
