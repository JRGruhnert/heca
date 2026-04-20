from dataclasses import dataclass, field
from tqdm import trange

from conf.properties import get_property_set
from conf.properties.v1.area import CalvinAreaConfig
from conf.entities import tdp_to_tde
from conf.agents import get_skill_set
from hoopgn.buffer import BufferConfig
from hoopgn.environments.entities.entity import Entity
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.dense import Dense3EvaluatorConfig
from hoopgn.evaluators.evaluator import EvaluatorConfig
from hoopgn.evaluators.set_evaluator import SetEvaluatorConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from hoopgn.networks.mp_gnn import MPGNNConfig
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import (
    HoopGNPlot,
    HoopGNPlotterConfig,
)
from hoopgn.environments.properties.v1.area_state import AreaStateConfig
from hoopgn.experiments import select_experiment
from hoopgn.agents.branches.hoopgn_agent import HoopGNSkill, HoopGNSkillConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.observation.td_properties import TDProperties
from hoopgn.plotters.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DMode,
)


skills = get_skill_set("srpb")
properties = get_property_set("srpb")
checkpoint_path = "checkpoints/explain/blue/best.pt"


@dataclass
class SpawnAreaPlotterConfig(HoopGNPlotterConfig):
    title: str = "Spawn Area Plot"
    name: str = "spawn_area"
    subdir: str = "spawn_area"
    agent: HoopGNSkillConfig = HoopGNSkillConfig(
        network=MPGNNConfig(
            dim_skill=len(skills),
            dim_state=len(properties),
            checkpoint_path=checkpoint_path,
        ),
        properties=properties,
        skills=skills,
        buffer=BufferConfig(size=100),
    )
    experiment: ExperimentConfig = NoiseExperimentConfig(
        environment=CalvinEnvironmentConfig(),
        evaluator=Dense3EvaluatorConfig(
            states_eval=properties,
            states_network=properties,
        ),
        p_skip=0.0,
        p_rand=0.0,
        min_steps=len(skills),
    )
    evaluator: EvaluatorConfig = SetEvaluatorConfig(
        success_reward=1.0,
        states_eval=properties,
        states_network=properties,
    )
    area: AreaStateConfig = CalvinAreaConfig()
    entity3d: Entity3DHelperConfig = Entity3DHelperConfig(
        mode=Entity3DMode.DIFFERENCE,
    )


class SpawnAreaPlotter(HoopGNPlot):
    def __init__(self, config: SpawnAreaPlotterConfig):
        self.config = config
        self.experiment = select_experiment(config.experiment)
        self.evaluator = select_evaluator(config.evaluator)
        self.entities = select_entities(config.entities)
        self.agent = HoopGNSkill(config.agent)
        self.entity3d = Entity3DHelper(config.entity3d)

    def is_different(self, current: TDProperties, goal: TDProperties):
        return not self.evaluator.is_equal(current, goal)

    def do_task(self, current: TDProperties, goal: TDProperties) -> bool:
        episode_ended = False
        done = False
        while not episode_ended:
            skill = self.agent.act(current, goal)
            # expl_actor, _ = self.agent.explain(current, goal, skill)
            current, reward, done, episode_ended = self.experiment.step(skill)
            if self.agent.feedback(reward, done, episode_ended):
                self.agent.buffer.clear()

        return done

    def plot_content(self):
        for i in trange(
            self.config.agent.buffer.size,
            desc="Collecting samples for explanation",
        ):
            c, g = self.experiment.sample_task()
            for e in self.entities:
                self.entity3d.set_entity(
                    entity=e,
                    current=tdp_to_tde(c, e),
                    goal=tdp_to_tde(g, e),
                    different=self.is_different(c, g),
                    solved=self.do_task(c, g),
                )

        self.entity3d.show_entities()
        self.entity3d.show_areas(self.config.area)
        self.entity3d.show_edges()
        self.entity3d.finish()
