import abc
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import torch
from heca.conditions.pair import ConPair
from heca.data.data import DCEntity, DCScene, TDImage
from heca.data.entity import Entity
from heca.misc.base import Persistable
from heca.scenes.scene import Scene, SceneFeedback
from heca.image_encoders.dino_encoder import DinoEncoder
from heca.image_encoders.image_encoder import ImageEncoder
from heca.image_encoders.molmo_encoder import MolmoEncoder


class ExpertModel(Persistable, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Persistable.Config):
        scene: Scene.Config
        kp_extraction: ImageEncoder.Config = DinoEncoder.Config()
        state_extraction: ImageEncoder.Config = MolmoEncoder.Config()
        score_threshold: float = 0.5
        virtual_term_rand: float = 0.5

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene, auto_load=False)
        self.act_virtual = False
        self._force_recompute = False
        self._use_gt = True
        self._fit_rotation = True

    @cached_property
    def entities(self) -> dict[str, Entity]:
        ents = {
            label: entity
            for label, entity in self.scene.entities.items()
            if label in self.tps
        }
        if not self._fit_rotation:
            for entity in ents.values():
                entity.cfg.fit_rotation = False
        return ents

    def virtual(self) -> "ExpertModel":
        self.act_virtual = True
        return self

    def real(self) -> "ExpertModel":
        self.act_virtual = False
        return self

    def force_recompute(self) -> "ExpertModel":
        self._force_recompute = True
        return self

    def use_fit_rotation(self, flag: bool) -> "ExpertModel":
        """Set the rotation mode for this agent's entities.

        A single per-agent choice: either all entities fit rotation or none.
        The flag is applied when the entities are built, so it can be set
        before the policy is loaded.
        """
        self._fit_rotation = flag
        return self

    def use_gt(self, flag: bool) -> "ExpertModel":
        self._use_gt = flag
        if not flag:
            self.scene.load()
            self.kp_extractor = ImageEncoder.get(self.cfg.kp_extraction)
            self.ste_extractor = ImageEncoder.get(self.cfg.state_extraction)
            self.kp_extractor.prepare_for_scene(self.cfg.scene)
            self.ste_extractor.prepare_for_scene(self.cfg.scene)
        return self

    @cached_property
    def tps(self) -> set[str]:
        raise NotImplementedError

    def from_image(self, image: TDImage) -> dict[str, DCEntity]:
        kps3d, _, kp_scores = self.kp_extractor.extract_poses(image)
        states, state_scores = self.ste_extractor.extract_states(image)
        # extras = self.extras_extractor.extract_extra(image)
        extras = np.zeros(
            0
        )  # TODO: implement extra calculation for prismatic and revoulte
        # Sanity check on dimensions
        assert kps3d.shape[1] == len(self.scene.entities)

        dc_entities: dict[str, DCEntity] = {}
        for idx, (key, entity) in enumerate(self.scene.entities.items()):
            pose, ste = self.get_entity_pose_and_state(
                kps3d[:, idx], kp_scores[:, idx], states[:, idx], state_scores[:, idx]
            )
            extra = extras[:, idx]
            dc_entities[key] = entity.dc_from_parsed(pose, extra, ste)
        return dc_entities

    def make_scene(self, scene: DCScene, image: TDImage) -> DCScene:
        if self._use_gt:
            return scene
        else:
            return DCScene(self.from_image(image), scene.extras)

    def get_entity_pose_and_state(
        self,
        poses: torch.Tensor,
        poses_scores: torch.Tensor,
        states: torch.Tensor,
        state_scores: torch.Tensor,
    ) -> tuple[np.ndarray, np.ndarray]:
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
        return pose.numpy(), state.numpy()

    @cached_property
    def conditions(self) -> ConPair:
        raise NotImplementedError

    def valid_task(self, x: DCScene, y: DCScene) -> bool:
        for label, entity in self.entities.items():
            x_valid = entity.score_single(
                x.get(label).value,
                self.conditions.pre.models[label].get_parameters(),
            )
            y_valid = entity.score_single(
                y.get(label).value,
                self.conditions.post.models[label].get_parameters(),
            )
            if not (x_valid and y_valid):
                return False
        return True

    def _act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        raise NotImplementedError

    def _act_virt(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        raise NotImplementedError

    def act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        if self.act_virtual:
            return self._act_virt(x, y)

        else:
            return self._act(x, y)

    @classmethod
    def load_dir(cls, cfg: "ExpertModel.Config") -> Path:
        """
        cls.root / cfg.scene.folder / cfg.scene.label / cfg.scene.tag
        """
        scene = cfg.scene
        tag = scene.load_tag or scene.tag
        path = cls.instance_dir(scene, scene.folder) / tag / "experts" / cfg.tag
        path.mkdir(parents=True, exist_ok=True)
        return path

    @classmethod
    def save_dir(cls, cfg: "ExpertModel.Config") -> Path:
        """
        cls.root / cfg.scene.folder / cfg.scene.label / cfg.scene.tag
        """

        scene = cfg.scene
        path = cls.instance_dir(scene, scene.folder) / scene.tag / "experts" / cfg.tag
        path.mkdir(parents=True, exist_ok=True)
        return path
