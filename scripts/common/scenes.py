from typing import Iterator

from conf.scenes import SCENE_MODULES
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene


def iter_agents() -> Iterator[ExpertModel.Config]:
    """Yield every agent config from the given (or all) scene modules."""
    for mod in SCENE_MODULES:
        yield from mod.agents


def iter_scene_configs() -> Iterator[Scene.Config]:
    """Yield each unique scene config from the given (or all) scene modules."""
    seen = set()
    for mod in SCENE_MODULES:
        for agent_cfg in mod.agents:
            assert isinstance(agent_cfg, ExpertModel.Config)
            if agent_cfg.scene.tag in seen:
                continue
            seen.add(agent_cfg.scene.tag)
            yield agent_cfg.scene


def agents_by_scene() -> list[tuple[Scene.Config, list[ExpertModel.Config]]]:
    result = []
    for mod in SCENE_MODULES:
        result.append((mod.agents[0].scene, list(mod.agents)))
    return result


def find_model(scene_tag: str, model_tag: str) -> ExpertModel.Config:
    """Return the first agent config matching the filters, or ``None``."""
    for cfg in iter_agents():
        if cfg.scene.tag != scene_tag:
            continue
        if cfg.tag != model_tag:
            continue
        return cfg
    raise ValueError


def find_scene_models(scene_tag: str) -> list[ExpertModel.Config]:
    """Return the first agent config matching the filters, or ``None``."""
    for scene_cfg, model_cfgs in agents_by_scene():
        if scene_cfg.tag == scene_tag:
            return model_cfgs
    raise ValueError


def find_scene_config(scene_tag: str) -> Scene.Config:
    """Return the first scene config with the given tag, or ``None``."""
    for scene_cfg in iter_scene_configs():
        if scene_cfg.tag == scene_tag:
            return scene_cfg
    raise ValueError
