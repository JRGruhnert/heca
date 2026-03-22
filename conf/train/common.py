from cli.commands.explain import ExplainManagerConfig
from conf.common import (
    agent_config,
    environment_config,
    experiment_config,
    logger_config,
    storage_config,
    network_config,
    evaluator_config,
)
from src.modules.buffer import BufferConfig
from src.modules.logger import LogMode


def get_explain_config(
    skill_set_tag: str,
    state_set_tag: str,
    is_baseline: bool = False,
    prefix_tag: str = "",
    checkpoint_name: str | None = None,
    p_empty: float = 0.0,
    p_rand: float = 0.0,
) -> ExplainManagerConfig:
    network = network_config(
        checkpoint_name,
        is_baseline,
        explain_mode=False,
    )
    return ExplainManagerConfig(
        agent=agent_config(
            network,
            False,
        ),
        buffer=BufferConfig(steps=1024),
        logger=logger_config(
            LogMode.WANDB,
            network.name,
            prefix_tag,
            state_set_tag,
            skill_set_tag,
        ),
        storage=storage_config(
            network.name,
            prefix_tag,
            state_set_tag,
            skill_set_tag,
        ),
        experiment=experiment_config(
            p_empty,
            p_rand,
        ),
        environment=environment_config(
            environment_tag="calvin",
            render=False,
        ),
        evaluator=evaluator_config("dense3"),
    )
