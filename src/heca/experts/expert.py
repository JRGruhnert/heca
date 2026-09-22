import abc
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import Sequence

import numpy as np
from heca.data.pair import ConPair
from heca.data.data import DCEntity, DCScene, TDImage
from heca.data.entity import Entity
from heca.data.reference import Reference
from heca.image_encoders.prior import fill_gaps
from heca.misc import logger
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

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.scene = Scene.get(self.cfg.scene, auto_load=False)
        self.act_virtual = False
        self._force_recompute = False
        self._use_gt = True

    @cached_property
    def entities(self) -> dict[str, Entity]:
        ents = {
            label: entity
            for label, entity in self.scene.entities.items()
            if label in self.tps
        }
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

    def reset_tracking(self):
        """Start a new episode: no belief carries over from the previous one."""
        if self._use_gt:
            return self
        self.kp_extractor.reset_tracking()
        return self

    def read_image(
        self, image: TDImage
    ) -> tuple[dict[str, DCEntity], dict[str, float]]:
        poses = self.kp_extractor.extract_poses(image)
        states = self.ste_extractor.extract_states(image)
        references = self.scene.references

        entities: dict[str, DCEntity] = {}
        confidence: dict[str, float] = {}
        for label, entity in self.scene.entities.items():
            keypoint = poses[label]
            confidence[label] = keypoint.score
            reference = references[label][Reference.POSITION].xyz
            position = keypoint.xyz
            if keypoint.score < self.cfg.score_threshold:
                logger.warning(
                    f"{label}: keypoint confidence {keypoint.score:.3f} below "
                    f"{self.cfg.score_threshold}, falling back to its reference"
                )
                position = reference

            positions = {
                "current": position,
                "reference": reference,
                **{
                    name: references[label][name].xyz for name in entity.reference_names
                },
            }
            extra = entity.extra_from_references(label, positions)
            ste = np.array([states[label][0] if label in states else 0])
            pose = np.concatenate(
                (self.scene.normalize_position(position), np.zeros(entity.rot_dim))
            )
            entities[label] = entity.dc_from_parsed(pose, extra, ste)
        return entities, confidence

    def from_image(self, image: TDImage) -> dict[str, DCEntity]:
        return self.read_image(image)[0]

    def encode_episode(
        self, images: Sequence[TDImage], scenes: Sequence[DCScene]
    ) -> list[DCScene]:
        self.reset_tracking()
        frames: list[dict[str, DCEntity]] = []
        confidence: list[dict[str, float]] = []
        for image in images:
            frame, sure = self.read_image(image)
            frames.append(frame)
            confidence.append(sure)

        for label, entity in self.scene.entities.items():
            weak = [
                index
                for index, sure in enumerate(confidence)
                if sure[label] < self.cfg.score_threshold
            ]
            if not weak or len(weak) == len(frames):
                if weak:
                    logger.warning(
                        f"{label}: nothing detected in any of the {len(frames)} "
                        "frames, its values are all reference positions"
                    )
                continue
            logger.warning(
                f"{label}: no detection in {len(weak)}/{len(frames)} frame(s), "
                "filling them from their neighbours"
            )
            values: list[np.ndarray | None] = [
                None if index in weak else frame[label].value
                for index, frame in enumerate(frames)
            ]
            filled = fill_gaps(values)
            extra_dim = entity.pose_dim - Entity.POS_DIM - entity.rot_dim
            for index in weak:
                pose = filled[index][: Entity.POS_DIM + entity.rot_dim]
                extra = filled[index][
                    Entity.POS_DIM
                    + entity.rot_dim : Entity.POS_DIM
                    + entity.rot_dim
                    + extra_dim
                ]
                frames[index][label] = entity.dc_from_parsed(
                    pose, extra, filled[index][-1:]
                )

        return [DCScene(frame, scene.extras) for frame, scene in zip(frames, scenes)]

    def make_scene(self, scene: DCScene, image: TDImage) -> DCScene:
        if self._use_gt:
            return scene
        else:
            return DCScene(self.from_image(image), scene.extras)

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

    def act(
        self, x: DCScene, y: DCScene, gated: bool = False
    ) -> tuple[DCScene, SceneFeedback]:
        if gated and self.act_virtual:
            z, fb = self.scene.gated_step_virt(x)
        elif self.act_virtual:
            z, fb = self.virtual_step(x, y)
        else:
            z, fb = self._act(x, y)
        return z, self.scene.count_option(fb)

    def _act(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        raise NotImplementedError

    def virtual_step(self, x: DCScene, y: DCScene) -> tuple[DCScene, SceneFeedback]:
        tdscene, tdimage, fb = self.scene.step_virt(
            x, y, self.conditions.target_entities
        )
        return self.make_scene(tdscene, tdimage), fb

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

    def fit_conditions(self):
        raise NotImplementedError
