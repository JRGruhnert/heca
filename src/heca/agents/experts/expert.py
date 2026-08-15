import abc
from dataclasses import dataclass
from functools import cached_property

import numpy as np
import torch
from heca.agents.agent import Agent
from heca.conditions.pair import ConPair
from heca.data.data import DCEntity, DCScene, TDImage
from heca.data.entity import Entity
from heca.scenes.scene import Scene, SceneFeedback
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
        super().__init__(cfg)
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene, auto_load=not cfg.use_gt)
        self.act_virtual = False
        if not self.cfg.use_gt:
            self.kp_extractor = ImageEncoder.get(self.cfg.kp_extraction)
            self.state_extractor = ImageEncoder.get(self.cfg.state_extraction)
            self.kp_extractor.prepare_for_scene(self.cfg.scene)
            self.state_extractor.prepare_for_scene(self.cfg.scene)

    @cached_property
    def entities(self) -> dict[str, Entity]:
        return {
            label: entity
            for label, entity in self.scene.entities.items()
            if label in self.tps()
        }

    def virtual(self) -> "ExpertAgent":
        self.act_virtual = True
        return self

    def tps(self) -> set[str]:
        raise NotImplementedError

    def from_image(self, image: TDImage) -> dict[str, DCEntity]:
        kps3d, kps2d, kp_scores = self.kp_extractor.extract_poses(image)
        states, state_scores = self.state_extractor.extract_states(image)

        # Sanity check on dimensions
        assert kps3d.shape[1] == len(self.scene.entities)

        dc_entities: dict[str, DCEntity] = {}
        for idx, (key, entity) in enumerate(self.scene.entities.items()):
            pos, rot, ste = self.get_entity_pose_and_state(
                kps3d[:, idx + 1],
                kp_scores[:, idx + 1],
                states[:, idx + 1],
                state_scores[:, idx + 1],
            )
            dc_entities[key] = entity.value_from_gt(pos, rot, ste)
        return dc_entities

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

    @cached_property
    def conditions(self) -> ConPair:
        raise NotImplementedError

    def valid_task(self, x: DCScene, y: DCScene) -> bool:
        for label, entity in self.entities.items():
            x_score, x_valid = entity.score_single(
                x.get(label).value,
                self.conditions.pre.models[label].get_parameters(),
            )
            yscore, y_valid = entity.score_single(
                y.get(label).value,
                self.conditions.post.models[label].get_parameters(),
            )
            if not (x_valid and y_valid):
                return False
        return True

    def _act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        raise NotImplementedError

    def act(self, x: DCScene, s: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        if self.act_virtual:
            if self.valid_task(x, s):
                return s.copy(), self.scene.virtual_evaluation(x, y)
            else:
                return x.copy(), SceneFeedback(
                    terminal=True, reward=0.0, truncated=False
                )
        else:
            return self._act(x, s)
