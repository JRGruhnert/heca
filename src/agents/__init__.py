from src.agents.human import HumanAgent, HumanAgentConfig
from src.agents.ppo import PPOAgent, PPOAgentConfig
from src.agents.search import SearchTreeAgent, SearchTreeAgentConfig


AGENT_BUILDERS = {
    PPOAgentConfig: lambda config, storage, buffer: PPOAgent(
        config,
        buffer,
        storage,
    ),
    SearchTreeAgentConfig: lambda config, storage, buffer: SearchTreeAgent(
        config,
        storage,
        buffer,
    ),
    HumanAgentConfig: lambda config, storage, buffer: HumanAgent(
        config,
        storage,
        buffer,
    ),
}


def register_agent(config_type, builder):
    AGENT_BUILDERS[config_type] = builder


def select_agent(config, storage, buffer):
    builder = AGENT_BUILDERS.get(type(config))
    if builder is None:
        for cfg_type, b in AGENT_BUILDERS.items():
            if isinstance(config, cfg_type):
                builder = b
                break
    if builder is None:
        raise ValueError(f"Unknown config type: {type(config)}")
    return builder(config, storage, buffer)
