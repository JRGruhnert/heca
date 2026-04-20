from conf.properties import get_property_set
from conf.skills import get_skill_set
from hoopgn.agents.branches.hoopgn_agent import HoopGNSkillConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.buffer import BufferConfig
from hoopgn.evaluators.dense import Dense3EvaluatorConfig
from hoopgn.networks.v1 import HoopgnV1Config
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from cli.cmd_train import TrainRunnerConfig


SKILL_TAG = "blue"
PROPERTY_TAG = "blue"

checkpoint_path = "checkpoints/explain/blue/best.pt"
skills = get_skill_set(SKILL_TAG)

properties = get_property_set(PROPERTY_TAG)


cfg = TrainRunnerConfig(
    skills=skills,
    properties=properties,
    agent=HoopGNSkillConfig(
        network=HoopgnV1Config(dim_skill=len(skills), dim_state=len(properties)),
        buffer=BufferConfig(size=5),
    ),
    experiment=NoiseExperimentConfig(
        environments=[CalvinEnvironmentConfig()],
        evaluator=Dense3EvaluatorConfig(
            states_eval=properties,
            states_network=properties,
        ),
        p_skip=0.0,
        p_rand=0.0,
        min_steps=len(skills),
    ),
)
