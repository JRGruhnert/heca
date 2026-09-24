from importlib import import_module
from types import ModuleType

SCENE_TAGS = (
    "scene0",
    "scene1",
    "scene2",
    "scene3",
    "scene4",
    "scene5",
    "scene6",
    "scene7",
    "scene8",
    "scene9",
    "scene10",
)


def scene_module(scene_tag: str) -> ModuleType:
    """The ``conf.experts`` module of one scene (imported on first use)."""
    if scene_tag not in SCENE_TAGS:
        raise ValueError(f"unknown scene {scene_tag!r}; known: {', '.join(SCENE_TAGS)}")
    return import_module(f"conf.experts.{scene_tag}")


def scene_modules() -> list[ModuleType]:
    """Every scene module, imported."""
    return [scene_module(scene_tag) for scene_tag in SCENE_TAGS]
