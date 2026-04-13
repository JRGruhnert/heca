from dataclasses import dataclass
import re
from tqdm import trange

from hoopgn.properties.states.area_state import AreaStateConfig
from hoopgn.experiments import select_experiment
from hoopgn.logger import LoggerConfig, Logger
from hoopgn.storage import Storage, StorageConfig
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.observation.td_parameters import TDParameters
from hoopgn.plots.entity_plot import Entity3DPlot
from hoopgn.properties.property import Property
from hoopgn.variables import BLUE, PINK, RED, SLIDE


@dataclass
class ExplainManagerConfig:
    agent: PPOAgentConfig
    logger: LoggerConfig
    storage: StorageConfig
    experiment: ExperimentConfig
    areas: AreaStateConfig
    eval_set: str


class ExplainScript:
    def __init__(self, config: ExplainManagerConfig):
        self.config = config
        self.storage = Storage(config.storage)
        self.logger = Logger(config.logger)
        self.experiment = select_experiment(config.experiment)
        self.agent = PPOAgent(config.agent, self.storage)

        self.relevant_objects = {
            RED: "block_red",
            BLUE: "block_blue",
            PINK: "block_pink",
            SLIDE: "slide",
        }
        self.object_pattern = re.compile(r"(red|blue|pink|slider)")
        # print(f"Checkpoint path: {self.config.agent.network.checkpoint_path}")
        match = self.object_pattern.search(
            self.config.agent.network.checkpoint_path or ""
        )
        if match:
            object_name = match.group(1)  # 'red', 'blue', 'pink', or 'slide'
        self.trained_object = self.relevant_objects[object_name]
        self.current_object = self.relevant_objects[self.config.eval_set]

        self.plot = Entity3DPlot(object_label=self.current_object)

        self.pos_state: Property = self.storage.get_property_by_name(
            f"{self.current_object}_position"
        )
        self.quat_state: Property = self.storage.get_property_by_name(
            f"{self.current_object}_rotation"
        )

    def is_different_on_start(self, current: TDParameters, goal: TDParameters):
        if self.pos_state is None or self.quat_state is None:
            raise ValueError("Position or rotation state not found in storage.")
        pos_eq = self.pos_state.evaluate(
            current[f"{self.current_object}_position"],
            goal[f"{self.current_object}_position"],
        )
        rot_eq = self.quat_state.evaluate(
            current[f"{self.current_object}_rotation"],
            goal[f"{self.current_object}_rotation"],
        )
        return not pos_eq

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

    def run_batch(self):
        """Collect experiences until batch is ready"""

        for i in trange(
            self.config.agent.batch_size, desc="Collecting samples for explanation"
        ):
            current, goal = self.experiment.sample_task()
            self.plot.set_entity(
                {
                    "current": current[f"{self.current_object}_position"].tolist(),
                    "goal": goal[f"{self.current_object}_position"].tolist(),
                },
                {
                    "current": current[f"{self.current_object}_rotation"].tolist(),
                    "goal": goal[f"{self.current_object}_rotation"].tolist(),
                },
                different_on_start=self.is_different_on_start(current, goal),
                solved_in_the_end=self.do_task(current, goal),
            )

        self.plot.show_objects()
        self.plot.show_areas(self.config.areas)
        self.plot.show_edges()
        self.plot.create(
            f"Sampled Start and Goal Positions of {self.trained_object} and {self.current_object}.",
            True,
            True,
            f"{self.storage.agent_saving_path(self.config.agent.network.label)}/plots/{self.trained_object}_s_{self.current_object}.png",
        )

    def run(self):
        """Main training loop"""
        metadata = self.experiment.metadata()
        metadata.update(self.agent.metadata())
        self.logger.initialize(metadata)
        self.run_batch()
        self.logger.log(self.agent.metrics())
        self.experiment.close()
