"""Shared scene / agent config loading helpers for the pipeline scripts."""

import copy
import dataclasses

from conf.scenes import SCENE_MODULES


def iter_agents(modules=None):
    """Yield every agent config from the given (or all) scene modules."""
    for mod in modules if modules is not None else SCENE_MODULES:
        yield from mod.agents


def iter_scene_configs(modules=None):
    """Yield each unique scene config from the given (or all) scene modules."""
    seen = set()
    for mod in modules if modules is not None else SCENE_MODULES:
        for agent_cfg in mod.agents:
            scene_cfg = agent_cfg.scene
            if scene_cfg.tag in seen:
                continue
            seen.add(scene_cfg.tag)
            yield scene_cfg


def agents_by_scene(modules=None) -> dict[str, list]:
    """Return ``{scene_tag: [agent_config, ...]}`` across the scene modules."""
    result = {}
    for mod in modules if modules is not None else SCENE_MODULES:
        result[mod.agents[0].scene.tag] = list(mod.agents)
    return result


def find_agent(scene_tag=None, tag=None):
    """Return the first agent config matching the filters, or ``None``."""
    for cfg in iter_agents():
        if scene_tag is not None and cfg.scene.tag != scene_tag:
            continue
        if tag is not None and cfg.tag != tag:
            continue
        return cfg
    return None


def to_ee(agent_cfg):
    """Return an ``ee_`` variant of an agent config, computed on the fly.

    The variant is a copy of ``agent_cfg`` with the tag prefixed ``ee_`` and
    the ``ee_init`` / ``ee_target`` entries removed from ``gt_frames``. The
    policy (and scene) are copied so the original config is left untouched.
    """
    gt_frames = None
    if agent_cfg.gt_frames is not None:
        gt_frames = [
            [name for name in frame if name not in ("ee_init", "ee_target")]
            for frame in agent_cfg.gt_frames
        ]
    return dataclasses.replace(
        agent_cfg,
        tag=f"ee_{agent_cfg.tag}",
        scene=copy.deepcopy(agent_cfg.scene),
        gt_frames=gt_frames,
        policy=copy.deepcopy(agent_cfg.policy),
    )


def find_scene_config(scene_tag):
    """Return the first scene config with the given tag, or ``None``."""
    for scene_cfg in iter_scene_configs():
        if scene_cfg.tag == scene_tag:
            return scene_cfg
    return None
