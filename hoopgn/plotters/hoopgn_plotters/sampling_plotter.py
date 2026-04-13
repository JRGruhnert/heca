from dataclasses import dataclass
from typing import cast
from tqdm import trange

from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.set_evaluator import SetEvaluatorConfig
from hoopgn.observation import convert_parameters_to_entities
from hoopgn.observation.td_entity import TDEntity
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import HoopGNPlot, HoopGNPlotConfig
from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.experiments import select_experiment
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.observation.td_parameters import TDParameters
from hoopgn.plotters.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DMode,
)


@dataclass
class SpawnAreaPlotConfig(HoopGNPlotConfig):
    agent: PPOAgentConfig
    experiment: ExperimentConfig
    evaluator: SetEvaluatorConfig
    areas: AreaStateConfig
    entity3d: Entity3DHelperConfig = Entity3DHelperConfig(
        mode=Entity3DMode.DIFF,
    )


class SpawnAreaPlot(HoopGNPlot):
    def __init__(self, config: SpawnAreaPlotConfig):
        self.config = config
        self.experiment = select_experiment(config.experiment)
        self.evaluator = select_evaluator(config.evaluator)
        self.agent = PPOAgent(config.agent)
        self.entity3d = Entity3DHelper(config.entity3d)

    def is_different(self, current: TDParameters, goal: TDParameters):
        return not self.evaluator.is_equal(current, goal)

    def do_task(self, current: TDParameters, goal: TDParameters) -> bool:
        episode_ended = False
        done = False
        while not episode_ended:
            skill = self.agent.act(current, goal)
            # expl_actor, _ = self.agent.explain(current, goal, skill)
            current, reward, done, episode_ended = self.experiment.step(skill)
            if self.agent.feedback(reward, done, episode_ended):
                self.agent.buffer.clear()

        return done

    def run(self):
        for i in trange(
            self.config.agent.buffer.steps,
            desc="Collecting samples for explanation",
        ):
            c, g = self.experiment.sample_task()
            es, ges, ces = convert_parameters_to_entities(c, g)
            for e in es:
                self.entity3d.set_entity(
                    e,
                    cast(TDEntity, ces[e.config.label]),
                    cast(TDEntity, ges[e.config.label]),
                    self.is_different(c, g),
                    self.do_task(c, g),
                )

        self.entity3d.show_entities()
        self.entity3d.show_areas(self.config.areas)
        self.entity3d.show_edges()
