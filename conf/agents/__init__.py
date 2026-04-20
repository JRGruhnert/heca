from hoopgn import logger
from hoopgn.agents.agent import SkillConfig
from conf.agents import v1


def get_skill_set(
    tag: str,
    vtag: str = "v1",
) -> list[SkillConfig]:
    assert vtag in ["v1", "v2"], f"Unsupported version tag: {vtag}"
    if vtag == "v1":
        return v1.get_set(tag)
    elif vtag == "v2":
        logger.warning("Using v2 skill set. This is a work in progress.")
    # Add more versions here as needed
    raise ValueError(f"Unsupported version tag: {vtag}")
