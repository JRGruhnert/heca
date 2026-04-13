from hoopgn.environments.calvin import CalvinEnvironmentConfig
from src.experiments.skill_check import SkillCheckExperimentConfig
from hoopgn.evaluators.skill import SkillEvaluatorConfig
from hoopgn.evaluators.sparse import SparseEvaluatorConfig
from hoopgn.logger import LogMode, LoggerConfig
from hoopgn.storage import StorageConfig
from hoopgn.variables import SET_SRPB

mode = LogMode.TERMINAL
render = False
tag = "eval"

config = SkillEvalConfig(
    tag=tag,
    iterations=100,
    logger=LoggerConfig(
        mode=mode,
        wandb_tag=tag,
    ),
    storage=StorageConfig(
        used_skills=SET_SRPB,
        used_states=SET_SRPB,
        tag=tag,
        network="none",
    ),
    experiment=SkillCheckExperimentConfig(
        evaluator=SkillEvaluatorConfig(),
        max_sample_attempts=20,
        sample_with_precons=True,
    ),
    environment=CalvinEnvironmentConfig(render=render),
    evaluator=SparseEvaluatorConfig(
        step_reward=0.0,
        success_reward=0.0,
    ),
)
