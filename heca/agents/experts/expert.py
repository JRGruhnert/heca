import abc
from dataclasses import dataclass

import torch
from heca.agents.agent import Agent, EESate
from heca.scenes.scene import Scene
from heca.image_encoders.dino_encoder import DinoEncoder
from heca.image_encoders.image_encoder import ImageEncoder
from heca.image_encoders.molmo_encoder import MolmoEncoder
from heca.misc.td import (
    TDEntity,
    TDImage,
    TDScene,
    make_abs_and_rel_td_entity,
)


class ExpertAgent(Agent, abc.ABC):
    @dataclass(kw_only=True)
    class Config(Agent.Config):
        subroot: str = "experts"
        scene: Scene.Config
        kp_extraction: ImageEncoder.Config = DinoEncoder.Config()
        state_extraction: ImageEncoder.Config = MolmoEncoder.Config()
        score_threshold: float = 0.5
        use_gt: bool = False

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = EESate.IDLE

        self.scene = Scene.get(self.cfg.scene, auto_load=not cfg.use_gt)

        if not self.cfg.use_gt:
            self.kp_extractor = ImageEncoder.get(self.cfg.kp_extraction)
            self.state_extractor = ImageEncoder.get(self.cfg.state_extraction)
            self.kp_extractor.prepare_for_scene(self.cfg.scene)
            self.state_extractor.prepare_for_scene(self.cfg.scene)

    def required_scenes(self) -> list[Scene.Config]:
        return [self.cfg.scene]

    def from_image(self, image: TDImage) -> TDScene:
        kps3d, kps2d, kp_scores = self.kp_extractor.extract_entities(image)
        states, state_scores = self.state_extractor.extract_states(image, kps2d)

        # Sanity check on dimensions
        assert kps3d.shape[1] == len(self.scene.entities) + 1  # ee at index 0

        td_entities: dict[str, TDEntity] = {}
        c_pos, c_rot, c_state = self.get_entity_pose_and_state(
            kps3d[:, 0],
            kp_scores[:, 0],
            states[:, 0],
            state_scores[:, 0],
        )
        for idx, entity in enumerate(self.scene.entities):
            pos, rot, state = self.get_entity_pose_and_state(
                kps3d[:, idx + 1],
                kp_scores[:, idx + 1],
                states[:, idx + 1],
                state_scores[:, idx + 1],
            )

            td_abs, td_rel = make_abs_and_rel_td_entity(
                position=pos,
                rotation=rot,
                state=state,
                ee_pos=c_pos,
                ee_rot=c_rot,
            )
            td_entities[entity.cfg.label] = td_abs
            td_entities[f"{entity.cfg.label}_rel"] = td_rel
        td_entities["ee"] = TDEntity(position=c_pos, rotation=c_rot, state=c_state)
        return TDScene(td_entities)

    def get_entity_pose_and_state(
        self,
        poses: torch.Tensor,
        poses_scores: torch.Tensor,
        states: torch.Tensor,
        state_scores: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        return pose[:3], pose[3:7], state

    # def from_images(self, td_image: TDImage) -> TDScene:
    #     td_images = TDSceneImages({"front": td_image})
    #     all_kps_3d = []
    #     all_states = []
    #     all_kp_scores = []
    #     all_state_scores = []

    #     all_cursor_kps_3d = []
    #     all_cursor_states = []
    #     all_cursor_kps_scores = []
    #     all_cursor_state_scores = []

    #     for cam, image in td_images.items():
    #         assert isinstance(image, TDImage)
    #         if cam.endswith("cursor"):
    #             kp3d_cursor, kp_cursor, kp_cursor_score = (
    #                 self.kp_extractor.extract_entities(image)
    #             )
    #             kp_cursor_state, cursor_state_scores = (
    #                 self.state_extractor.extract_states(image, kp_cursor)
    #             )

    #             all_cursor_kps_3d.append(kp3d_cursor)
    #             all_cursor_states.append(kp_cursor_state)
    #             all_cursor_kps_scores.append(kp_cursor_score)
    #             all_cursor_state_scores.append(cursor_state_scores)

    #         else:
    #             kps3d, kps2d, kp_scores = self.kp_extractor.extract_entities(image)
    #             states, state_scores = self.state_extractor.extract_states(image, kps2d)

    #             all_kps_3d.append(kps3d)
    #             all_states.append(states)
    #             all_kp_scores.append(kp_scores)
    #             all_state_scores.append(state_scores)

    #     kps_3d_stack = torch.stack(all_kps_3d)  # (C, K, D)
    #     states_stack = torch.stack(all_states)  # (C, K, S)
    #     kp_3d_scores_stack = torch.stack(all_kp_scores)  # (C, K)
    #     state_scores_stack = torch.stack(all_state_scores)  # (C, K)

    #     # For cursor
    #     cursor_kps_3d_stack = torch.stack(all_cursor_kps_3d)  # (C, 1, D)
    #     cursor_states_stack = torch.stack(all_cursor_states)  # (C, 1, S)
    #     cursor_kp_scores_stack = torch.stack(all_cursor_kps_scores)  # (C, 1)
    #     cursor_state_scores_stack = torch.stack(all_cursor_state_scores)  # (C, 1)

    #     # Sanity check on dimensions
    #     assert kps_3d_stack.shape[1] == len(self.scene.entities)

    #     td_entities: dict[str, TDEntity] = {}
    #     c_pos, c_rot, c_state = self.get_entity_pose_and_state(
    #         cursor_kps_3d_stack[:, 0],
    #         cursor_kp_scores_stack[:, 0],
    #         cursor_states_stack[:, 0],
    #         cursor_state_scores_stack[:, 0],
    #     )
    #     for idx, entity in enumerate(self.scene.entities):
    #         pos, rot, state = self.get_entity_pose_and_state(
    #             kps_3d_stack[:, idx],
    #             kp_3d_scores_stack[:, idx],
    #             states_stack[:, idx],
    #             state_scores_stack[:, idx],
    #         )

    #         td_abs, td_rel = make_abs_and_rel_td_entity(
    #             position=pos,
    #             rotation=rot,
    #             state=state,
    #             cursor_pos=c_pos,
    #             cursor_rot=c_rot,
    #         )
    #         td_entities[entity.cfg.label] = td_abs
    #         td_entities[f"{entity.cfg.label}_rel"] = td_rel
    #     td_entities["cursor"] = TDEntity(position=c_pos, rotation=c_rot, state=c_state)
    #     return TDScene(td_entities)
