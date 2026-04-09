from cli.commands.explain import ExplainManagerConfig
from conf.common import (
    agent_config,
    environment_config,
    experiment_config,
    logger_config,
    skill_configs,
    storage_config,
    network_config,
    evaluator_config,
)
from src.buffer import BufferConfig
from src.logger import LogMode
from src.storage import Storage


def get_explain_config(
    skill_set_tag: str,
    state_set_tag: str,
    checkpoint_name: str,
    is_gnn: bool = True,
    prefix_tag: str = "",
) -> ExplainManagerConfig:
    states = skill_configs(skill_set_tag, state_set_tag)
    skills = skill_configs(skill_set_tag, state_set_tag)
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
        agent=agent_config(
            network,
            True,
        ),
        buffer=BufferConfig(steps=16),
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
        ),
        environment=environment_config(
            environment_tag="calvin",
            render=False,
        ),
        evaluator=evaluator_config("dense3"),
    )
