from dataclasses import dataclass

from heca.agents.hecas.hoopgn_agent import HoopGNSkill, HoopGNSkillConfig

from heca.experiments import select_experiment
from heca.experiments.experiment import ExperimentConfig
from heca.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class ExplainRunnerConfig(HoopGNRunnerConfig):
    agent: HoopGNSkillConfig
    experiment: ExperimentConfig


class ExplainRunner(HoopGNRunner):
    def __init__(self, config: ExplainRunnerConfig):
        super().__init__(config)
        self.config = config
        self.experiment = select_experiment(config.experiment)
        self.agent = HoopGNSkill(config.agent)

    def run(self):
        obs, goal = self.experiment.sample_task()
        episode_ended = False
        while not episode_ended:
            skill = self.agent.act(obs, goal)
            actor_expl, critic_expl = self.agent.explain(obs, goal, skill)
