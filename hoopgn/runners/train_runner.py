from dataclasses import dataclass

from hoopgn.experiments import select_experiment
from hoopgn.skills.branches.hoopgn.hoopgn_skill import HoopGNSkill, HoopGNSkillConfig
from hoopgn.experiments.experiment import ExperimentConfig
from hoopgn.runners.runner import HoopGNRunner, HoopGNRunnerConfig


@dataclass
class TrainRunnerConfig(HoopGNRunnerConfig):
    skill: HoopGNSkillConfig
    experiment: ExperimentConfig


class TrainRunner(HoopGNRunner):
    def __init__(self, config: TrainRunnerConfig):
        super().__init__(config)
        self.config = config
        self.experiment = select_experiment(config.experiment)
        self.skill = HoopGNSkill(config.skill)

    def collect_batch(self) -> bool:
        while True:
            obs, goal = self.experiment.sample_task()
            episode_ended = False
            while not episode_ended:
                skill = self.skill.act(obs, goal)
                obs, reward, done, episode_ended = self.experiment.step(skill)
                if self.skill.feedback(reward, done, episode_ended):
                    return True

    def run(self):
        stop = False
        while not stop:
            self.collect_batch()
            stop = self.skill.learn()
        self.experiment.close()
