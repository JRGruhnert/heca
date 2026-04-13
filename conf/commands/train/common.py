from cli.cmd_train import TrainRunnerConfig
from conf.common import (
    ppo_default_config,
    experiment_config,
    logger_config,
    storage_config,
    network_config,
)
from conf.property_sets import OBJECT_SETS
from conf.skill_sets import SKILL_SETS
from hoopgn.logger import LogMode


def get_train_config(
    skill_set_tag: str,
    state_set_tag: str,
    is_gnn: bool = False,
    prefix_tag: str = "",
    checkpoint_name: str | None = None,
    p_empty: float = 0.0,
    p_rand: float = 0.0,
    batch_size: int = 1024,
    log_mode: LogMode = LogMode.WANDB,
) -> TrainRunnerConfig:
    skills = SKILL_SETS[skill_set_tag]
    states = OBJECT_SETS[state_set_tag]
    network = network_config(
        is_gnn=is_gnn,
        checkpoint_name=checkpoint_name,
        explain_mode=False,
        skill_count=len(skills),
        state_count=len(states),
    )
    return TrainRunnerConfig(
        agent=ppo_default_config(
            network=network,
            eval=False,
            batch_size=batch_size,
        ),
        logger=logger_config(
            log_mode,
            network.label,
            prefix_tag,
            state_set_tag,
            skill_set_tag,
        ),
        storage=storage_config(
            prefix_tag,
            state_set_tag,
            skill_set_tag,
        ),
        experiment=experiment_config(
            p_empty,
            p_rand,
            min_steps=len(skills),
            skill_set_tag=skill_set_tag,
            state_eval_tag=state_set_tag,
            state_network_tag=state_set_tag,
            environment_tag="calvin",
            evaluator_tag="dense3",
        ),
    )
