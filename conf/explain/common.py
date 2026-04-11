from cli.commands.explain import ExplainManagerConfig
from conf.common import (
    ppo_default_config,
    experiment_config,
    logger_config,
    storage_config,
    network_config,
)
from conf.property_sets import OBJECT_SETS
from conf.skill_sets import SKILL_SETS
from src.logger import LogMode


def get_explain_config(
    skill_set_tag: str,
    state_set_tag: str,
    checkpoint_name: str,
    is_gnn: bool = True,
    prefix_tag: str = "",
) -> ExplainManagerConfig:
    skills = SKILL_SETS[skill_set_tag]
    states = OBJECT_SETS[state_set_tag]
    storage = storage_config(
        prefix_tag,
        state_set_tag,
        skill_set_tag,
    )

    network = network_config(
        is_gnn,
        checkpoint_name,
        explain_mode=True,
        skill_count=len(skills),
        state_count=len(states),
    )
    return ExplainManagerConfig(
        agent=ppo_default_config(
            network=network,
            batch_size=16,
            eval=True,
        ),
        logger=logger_config(
            LogMode.TERMINAL,
            network.label,
            prefix_tag,
            state_set_tag,
            skill_set_tag,
        ),
        storage=storage,
        experiment=experiment_config(
            0.0,
            0.0,
            min_steps=len(skills),
            skill_set_tag=skill_set_tag,
            state_eval_tag=state_set_tag,
            state_network_tag=state_set_tag,
            environment_tag="calvin",
            evaluator_tag="dense3",
        ),
        eval_set=state_set_tag,
    )
