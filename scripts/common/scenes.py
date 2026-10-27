import dataclasses
import os
from collections.abc import Iterator, Sequence

from conf.scenes import SCENE_TAGS, scene_module
from heca.experts.expert import ExpertModel
from heca.scenes.scene import Scene


def scene_tags() -> tuple[str, ...]:
    """Every scene tag, without importing a single scene module."""
    return SCENE_TAGS


def selected_scenes(args) -> list[str]:
    """The scene tags a run trains on, one client each.

    ``scenes`` (a list) wins, then the single ``scene`` tag, else every scene.
    """
    if args.scenes:
        return list(args.scenes)
    return [args.scene] if args.scene else list(SCENE_TAGS)


def place_encoders(agents: list[ExpertModel.Config]) -> list[ExpertModel.Config]:
    """Honour HECA_KP_DEVICE / HECA_MOLMO_DEVICES for every agent of a scene.

    The two encoders of one process normally share ``hardware.device``. Several
    runs on one node can instead give every run's Molmo its own card and put all
    the small keypoint models on a single further card, which is what makes three
    arms fit on four 11 GB cards without spilling to host memory.
    """
    kp_device = os.environ.get("HECA_KP_DEVICE", "").strip()
    molmo_cards = os.environ.get("HECA_MOLMO_DEVICES", "").strip()
    backend = os.environ.get("HECA_KP_BACKEND", "").strip()
    if not kp_device and not molmo_cards and not backend:
        return agents
    for agent in agents:
        if backend and getattr(agent.kp_extraction, "backend", "dino") != backend:
            agent.kp_extraction = _backend_config(backend, agent.kp_extraction)
        if kp_device:
            agent.kp_extraction = dataclasses.replace(
                agent.kp_extraction, device=kp_device
            )
        if molmo_cards and hasattr(agent.state_extraction, "gpu_cards"):
            agent.state_extraction = dataclasses.replace(
                agent.state_extraction, gpu_cards=molmo_cards
            )
    return agents


def _backend_config(backend: str, base):
    """The keypoint config of one backend, keeping the card it was told to use."""
    device = getattr(base, "device", "")
    if backend == "roma":
        from heca.image_encoders.roma_encoder import RomaEncoder

        return RomaEncoder.Config(device=device)
    return dataclasses.replace(base, backend="dino", device=device)


def experts_for_scene(scene_tag: str) -> list[ExpertModel.Config]:
    """The agent configs of one scene; imports that scene only."""
    return place_encoders(list(scene_module(scene_tag).agents))


def scene_config(scene_tag: str) -> Scene.Config:
    """The scene config of one scene; imports that scene only."""
    return experts_for_scene(scene_tag)[0].scene


def select_encoder(
    tag: str, backbone: str = "", stride: int = 0, backend: str = "dino"
):
    """Point a scene's agents at one keypoint matcher."""
    agents = experts_for_scene(tag)
    base = agents[0].kp_extraction
    if not backbone and not stride and backend == getattr(
        base, "backend", "dino"
    ):
        return base

    if backend == "roma":
        from heca.image_encoders.roma_encoder import RomaEncoder

        chosen = RomaEncoder.Config(device=getattr(base, "device", ""))
    else:
        chosen = base
        changes: dict = {}
        if backbone:
            changes["backbone"] = backbone
        if stride:
            changes["stride"] = stride
        if changes:
            chosen = dataclasses.replace(chosen, **changes)
            chosen = dataclasses.replace(
                chosen, label=f"kp_{chosen.backbone.split('.')[0]}_{chosen.stride}"
            )
    for agent in agents:
        agent.kp_extraction = chosen
    return chosen


def agents_by_scene(
    tags: Sequence[str] = SCENE_TAGS,
) -> list[tuple[Scene.Config, list[ExpertModel.Config]]]:
    """``(scene, agents)`` for every given tag (all of them by default)."""
    return [(scene_config(tag), experts_for_scene(tag)) for tag in tags]


def iter_agents() -> Iterator[ExpertModel.Config]:
    """Yield every agent config, scene by scene."""
    for tag in SCENE_TAGS:
        yield from experts_for_scene(tag)


def iter_scene_configs() -> Iterator[Scene.Config]:
    """Yield the scene config of every scene."""
    for tag in SCENE_TAGS:
        yield scene_config(tag)


def find_model(scene_tag: str, model_tag: str) -> ExpertModel.Config:
    """Return the one agent config carrying both tags."""
    agents = experts_for_scene(scene_tag)
    for cfg in agents:
        if cfg.tag == model_tag:
            return cfg
    known = ", ".join(cfg.tag for cfg in agents)
    raise ValueError(f"no model {model_tag!r} in {scene_tag!r}; known: {known}")


def find_scene_models(scene_tag: str) -> list[ExpertModel.Config]:
    """Return every agent config of one scene."""
    return experts_for_scene(scene_tag)


def find_scene_config(scene_tag: str) -> Scene.Config:
    """Return the scene config with the given tag."""
    return scene_config(scene_tag)
