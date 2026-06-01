from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum
from heca.agents.agent import Agent
from heca.environment.scenes.scene import Scene
from heca.misc.td import TDEntity, TDImage, TDScene, TDSceneVision


class ExtractionMode(Enum):
    MOLMO = "molmo"
    DINO = "dino"
    GT = "gt"


class ExpertAgent(Agent):
    @dataclass(frozen=True, kw_only=True)
    class Query(Agent.Query):
        type: str

    @dataclass(kw_only=True)
    class Config(Agent.Config):
        scene: Scene.Query
        kp_extraction: ExtractionMode = ExtractionMode.MOLMO
        state_extraction: ExtractionMode = ExtractionMode.MOLMO

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @abstractmethod
    def act(self, x: TDScene, y: TDScene) -> TDScene:
        raise NotImplementedError()

    def required_scenes(self) -> list[Scene.Query]:
        return [self.cfg.scene]

    def from_vision(self, vision: TDSceneVision) -> TDScene:
        all_kps = []
        all_states = []
        all_infos = []
        all_masks = []
        all_scores = []

        for cam, td_image in vision.items():
            assert isinstance(td_image, TDImage)
            kps, states, info = self.extractors[cam].encode(td_image)
            all_kps.append(kps)
            all_states.append(states)
            all_infos.append(info)
            all_masks.append(info["kp_mask"])
            all_scores.append(info["state_scores"])

        kps_stack = torch.stack(all_kps)  # (C, K, D)
        kps_mask = torch.stack(all_masks)  # (C, K)
        scores = torch.stack(all_scores)  # (C, K)
        states_stack = torch.stack(all_states)  # (C, K, S)

        # Aggregate keypoints of different cameras
        num_kps = kps_stack.shape[1]
        final_kps = []
        final_states = []
        for k in range(num_kps):
            present = kps_mask[:, k] > 0
            if present.sum() == 0:
                # Not present in any camera
                final_kps.append(torch.full_like(kps_stack[0, k], float("nan")))
                final_states.append(torch.full_like(states_stack[0, k], float("nan")))
            elif present.sum() == 1:
                # Present in only one camera
                idx = present.nonzero(as_tuple=True)[0][0]
                final_kps.append(kps_stack[idx, k])
                final_states.append(states_stack[idx, k])
            else:
                # Present in multiple cameras
                # Mean for keypoints
                # Best score for states
                vals = kps_stack[present, k]
                final_kps.append(vals.mean(dim=0))
                valid_scores = scores[:, k].clone()
                valid_scores[~present] = float("-inf")
                idx = torch.argmax(valid_scores)
                final_states.append(states_stack[idx, k])
        final_kps = torch.stack(final_kps)  # (K, D)
        final_states = torch.stack(final_states)  # (K, S)

        td_entities: dict[str, TDEntity] = {}
        # Find the index of the end effector (ee) entity in self.entities
        num_present = min(final_kps.shape[0], final_states.shape[0], len(self.entities))
        for idx, entity in enumerate(self.scene.entities):
            if idx >= num_present:
                # No keypoint or descriptor for this entity, skip
                # Assuming keypoints are ordered the same as entities
                continue
            tg_kp = final_kps[idx]
            td_abs, td_rel = make_abs_and_rel_td_entity(
                position=tg_kp[:3],
                rotation=tg_kp[3:7],
                state=final_states[idx],
                cursor_pos=cursor_pos,
                cursor_rot=cursor_rot,
            )
            td_entities[f"{entity.cfg.label}_rel"] = td_rel
            td_entities[entity.cfg.label] = td_abs
        td_entities["cursor"] = TDEntity(
            position=cursor_pos,
            rotation=cursor_rot,
            state=cursor_state,
        )
        return TDScene(td_entities)
