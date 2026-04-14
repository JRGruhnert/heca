from dataclasses import dataclass, field
from tqdm import trange

from conf.entities.properties.area import CalvinAreaConfig
from conf.entity_set import properties_to_entities, tdp_to_tde
from conf.property_sets import get_property_set
from conf.skill_sets import get_skill_set
from hoopgn.buffer import BufferConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.evaluators import select_evaluator
from hoopgn.evaluators.dense3 import Dense3EvaluatorConfig
from hoopgn.evaluators.evaluator import EvaluatorConfig
from hoopgn.evaluators.set_evaluator import SetEvaluator, SetEvaluatorConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from hoopgn.networks.hoopgnv1 import HoopgnV1Config
from hoopgn.plotters.hoopgn_plotters.hoopgn_plotter import (
    HoopGNPlot,
    HoopGNPlotterConfig,
)
from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.experiments import select_experiment
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.observation.td_properties import TDProperties
from hoopgn.plotters.plots.entity_3d import (
    Entity3DHelper,
    Entity3DHelperConfig,
    Entity3DMode,
)
from hoopgn.storage import StorageConfig, select_entities

SKILLS = "srpb"
PROPERTIES_EVAL = "srpb"
PROPERTIES_NETWORK = "srpb"
checkpoint_path = "checkpoints/ppo_hoopgnv1_srpb.pt"
skills = get_skill_set(SKILLS)

properties_eval = get_property_set(PROPERTIES_EVAL)
properties_network = get_property_set(PROPERTIES_NETWORK)
properties = properties_network
entities = properties_to_entities(properties=properties)


@dataclass
class SpawnAreaPlotterConfig(HoopGNPlotterConfig):
    title: str = "Spawn Area Plot"
    name: str = "spawn_area"
    subdir: str = "spawn_area"
    entities: list = field(default_factory=lambda: entities)
    agent: PPOAgentConfig = PPOAgentConfig(
        network=HoopgnV1Config(dim_skill=len(skills), dim_state=len(properties)),
        storage=StorageConfig(
            skills=skills,
            states_eval=properties_eval,
            states_network=properties_network,
            checkpoint_path=checkpoint_path,
        ),
        buffer=BufferConfig(steps=100),
    )
    experiment: ExperimentConfig = NoiseExperimentConfig(
        environment=CalvinEnvironmentConfig(),
        evaluator=Dense3EvaluatorConfig(
            states_eval=properties_eval,
            states_network=properties_network,
        ),
        p_skip=0.0,
        p_rand=0.0,
        min_steps=len(skills),
    )
    evaluator: EvaluatorConfig = SetEvaluatorConfig(
        success_reward=1.0,
        states_eval=properties_eval,
        states_network=properties_network,
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
        self.agent = PPOAgent(config.agent)
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
            self.config.agent.buffer.steps,
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
