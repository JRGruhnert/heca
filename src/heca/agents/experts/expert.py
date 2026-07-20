import abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from heca.agents.agent import Agent
from heca.misc.data import DCEntity, DCScene, TDImage
from heca.misc.entity import Entity
from heca.scenes.scene import Scene
from heca.image_encoders.dino_encoder import DinoEncoder
from heca.image_encoders.image_encoder import ImageEncoder
from heca.image_encoders.molmo_encoder import MolmoEncoder


class ExpertAgent(Agent, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        scene: Scene.Config
        kp_extraction: ImageEncoder.Config = DinoEncoder.Config()
        state_extraction: ImageEncoder.Config = MolmoEncoder.Config()
        score_threshold: float = 0.5
        use_gt: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg

        self.scene = Scene.get(self.cfg.scene, auto_load=not cfg.use_gt)

        if not self.cfg.use_gt:
            self.kp_extractor = ImageEncoder.get(self.cfg.kp_extraction)
            self.state_extractor = ImageEncoder.get(self.cfg.state_extraction)
            self.kp_extractor.prepare_for_scene(self.cfg.scene)
            self.state_extractor.prepare_for_scene(self.cfg.scene)

    def required_scenes(self) -> list[Scene.Config]:
        return [self.cfg.scene]

    @cached_property
    def entities(self) -> dict[str, Entity]:
        return {entity.cfg.label: entity for entity in self.scene.entities}

    def from_image(self, image: TDImage) -> DCScene:
        kps3d, kps2d, kp_scores = self.kp_extractor.extract_entities(image)
        states, state_scores = self.state_extractor.extract_states(image, kps2d)

        # Sanity check on dimensions
        assert kps3d.shape[1] == len(self.scene.entities) + 1  # ee at index 0

        dc_entities: dict[str, DCEntity] = {}
        for idx, entity in enumerate(self.scene.entities):
            pos, rot, ste = self.get_entity_pose_and_state(
                kps3d[:, idx + 1],
                kp_scores[:, idx + 1],
                states[:, idx + 1],
                state_scores[:, idx + 1],
            )
            soh = entity.one_hot_from_idx_dc(ste.item())
            dc_entities[entity.cfg.label] = Entity.to_value(pos, rot, ste, soh)
        c_pos, c_rot, c_ste = self.get_entity_pose_and_state(
            kps3d[:, 0],
            kp_scores[:, 0],
            states[:, 0],
            state_scores[:, 0],
        )
        c_soh = entity.one_hot_from_idx_dc(c_ste.item())
        ee = Entity.to_value(c_pos, c_rot, c_ste, c_soh)
        return DCScene(ee, dc_entities)

    def get_entity_pose_and_state(
        self,
        poses: torch.Tensor,
        poses_scores: torch.Tensor,
        states: torch.Tensor,
        state_scores: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        present = poses_scores > self.cfg.score_threshold
        if present.sum() == 0:
            # Not present
            # TODO: handle missing keypoint, e.g. by interpolation or using a default value
            pass
        elif present.sum() == 1:
            # Present
            idxx = present.nonzero(as_tuple=True)[0][0]
            pose = poses[idxx]
            state = states[idxx]
        return pose[:3].numpy(), pose[3:7].numpy(), state.numpy()
