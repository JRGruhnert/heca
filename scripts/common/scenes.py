from collections.abc import Iterator, Sequence

from conf.scenes import SCENE_TAGS, scene_module
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene


def scene_tags() -> tuple[str, ...]:
    """Every scene tag, without importing a single scene module."""
    return SCENE_TAGS


def agents_for_scene(scene_tag: str) -> list[ExpertModel.Config]:
    """The agent configs of one scene; imports that scene only."""
    return list(scene_module(scene_tag).agents)


def scene_config(scene_tag: str) -> Scene.Config:
    """The scene config of one scene; imports that scene only."""
    return agents_for_scene(scene_tag)[0].scene


def agents_by_scene(
    tags: Sequence[str] = SCENE_TAGS,
) -> list[tuple[Scene.Config, list[ExpertModel.Config]]]:
    """``(scene, agents)`` for every given tag (all of them by default)."""
    return [(scene_config(tag), agents_for_scene(tag)) for tag in tags]


def iter_agents() -> Iterator[ExpertModel.Config]:
    """Yield every agent config, scene by scene."""
    for tag in SCENE_TAGS:
        yield from agents_for_scene(tag)


def iter_scene_configs() -> Iterator[Scene.Config]:
    """Yield the scene config of every scene."""
    for tag in SCENE_TAGS:
        yield scene_config(tag)


def find_model(scene_tag: str, model_tag: str) -> ExpertModel.Config:
    """Return the one agent config carrying both tags."""
    agents = agents_for_scene(scene_tag)
    for cfg in agents:
        if cfg.tag == model_tag:
            return cfg
    known = ", ".join(cfg.tag for cfg in agents)
    raise ValueError(f"no model {model_tag!r} in {scene_tag!r}; known: {known}")


def find_scene_models(scene_tag: str) -> list[ExpertModel.Config]:
    """Return every agent config of one scene."""
    return agents_for_scene(scene_tag)


def find_scene_config(scene_tag: str) -> Scene.Config:
    """Return the scene config with the given tag."""
    return scene_config(scene_tag)
