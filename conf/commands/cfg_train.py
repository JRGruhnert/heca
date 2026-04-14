from conf.entity_set import properties_to_entities
from hoopgn.agents.ppo import PPOAgentConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.buffer import BufferConfig
from hoopgn.evaluators.dense3 import Dense3EvaluatorConfig
from hoopgn.networks.hoopgnv1 import HoopgnV1Config
from hoopgn.storage import StorageConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from cli.cmd_train import TrainRunnerConfig

from conf.property_sets import get_property_set
from conf.skill_sets import get_skill_set


SKILLS = "blue"
PROPERTIES_EVAL = "blue"
PROPERTIES_NETWORK = "blue"
ENTITIES = "blue"


skills = get_skill_set(SKILLS)
properties_eval = get_property_set(PROPERTIES_EVAL)
properties_network = get_property_set(PROPERTIES_NETWORK)
properties = properties_network
entities = properties_to_entities(properties=properties)


cfg = TrainRunnerConfig(
    skills=skills,
    entities=entities,
    properties=properties,
    agent=PPOAgentConfig(
        network=HoopgnV1Config(dim_skill=len(skills), dim_state=len(properties)),
        storage=StorageConfig(
            skills=skills,
            states_eval=properties_eval,
            states_network=properties_network,
        ),
        buffer=BufferConfig(steps=5),
    ),
    experiment=NoiseExperimentConfig(
        environment=CalvinEnvironmentConfig(),
        evaluator=Dense3EvaluatorConfig(
            states_eval=properties_eval,
            states_network=properties_network,
        ),
        p_skip=0.0,
        p_rand=0.0,
        min_steps=len(skills),
    ),
)
