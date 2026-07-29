from dataclasses import dataclass

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.scenes.ogbench.scene import OGScene


class OGScene3(OGScene):
    @dataclass(kw_only=True)
    class Config(OGScene.Config):
        tag: str = "scene3"
        vis: bool = False

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg

    @property
    def description(self) -> str:
        raise NotImplementedError

    @property
    def entities(self) -> set[Entity]:
        ents = []
        return set([Entity.get(e) for e in ents])

    def to_dc_scene(self, obs: dict) -> DCScene:
        dc_entities: dict[str, DCEntity] = {}
        for entity in self.entities:
            if entity.cfg.label in [
                "button_0",
                "button_1",
            ]:  # hack cause _pos already used
                e_pos = obs[f"privileged_{entity.cfg.label}_pos_full"]
            else:
                e_pos = obs[f"privileged_{entity.cfg.label}_pos"]
            wxyz = obs[f"privileged_{entity.cfg.label}_quat"]
            e_rot = np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]], dtype=np.float32)
            e_ste = np.atleast_1d(obs[f"privileged_{entity.cfg.label}_state"])
            e_soh = entity.one_hot_from_idx_dc(e_ste)
            dc_entities[entity.cfg.label] = Entity.to_value(e_pos, e_rot, e_ste, e_soh)
        pos, rot, ste, soh = self.get_ee_dc(obs)
        ee = Entity.to_value(pos, rot, ste, soh)
        extras = self.get_extras(obs)
        return DCScene(ee, dc_entities, extras=extras)
