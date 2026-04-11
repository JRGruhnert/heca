from hoopgn.agents.agent import Agent
from hoopgn.agents.human import HumanAgent, HumanAgentConfig
from hoopgn.agents.ppo import PPOAgent, PPOAgentConfig


AGENT_BUILDERS = {
    PPOAgentConfig: lambda config, storage: PPOAgent(
        config,
        storage,
    ),
    HumanAgentConfig: lambda config, storage: HumanAgent(
        config,
        storage,
    ),
}


def register_agent(config_type, builder):
    AGENT_BUILDERS[config_type] = builder


def select_agent(config, storage) -> Agent:
    builder = AGENT_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in AGENT_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config, storage)
