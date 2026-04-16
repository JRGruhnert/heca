from hoopgn import logger
from hoopgn.environments.properties.property import PropertyConfig
from conf.properties import v1


def get_property_set(
    tag: str,
    vtag: str = "v1",
) -> list[PropertyConfig]:
    assert vtag in ["v1", "v2"], f"Unsupported version tag: {vtag}"
    if vtag == "v1":
        return v1.get_set(tag)
    elif vtag == "v2":
        logger.warning("Using v2 property set. This is a work in progress.")
    # Add more versions here as needed
    raise ValueError(f"Unsupported version tag: {vtag}")
