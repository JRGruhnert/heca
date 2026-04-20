from conf.properties import get_property_set
from conf.agents import get_skill_set
from hoopgn.agents.branches.hoopgn_agent import HoopGNSkillConfig
from hoopgn.buffer import BufferConfig
from hoopgn.environments.calvin import CalvinEnvironmentConfig
from hoopgn.evaluators.dense import Dense3EvaluatorConfig
from hoopgn.experiments.noise_experiment import NoiseExperimentConfig
from hoopgn.networks.mp_gnn import MPGNNConfig
from hoopgn.runners.explain_runner import ExplainRunnerConfig

SKILL_TAG = "blue"
PROPERTY_TAG = "blue"

checkpoint_path = "checkpoints/explain/blue/best.pt"
skills = get_skill_set(SKILL_TAG)

properties = get_property_set(PROPERTY_TAG)

cfg = ExplainRunnerConfig(
    skills=skills,
    properties=properties,
    agent=HoopGNSkillConfig(
        network=MPGNNConfig(
            dim_skill=len(skills),
            dim_state=len(properties),
            explain_mode=True,
        ),
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
