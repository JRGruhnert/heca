import numpy as np

from heca.misc.data import DCEntity


def load_demos():
    pass


def eval_state(a: DCEntity, b: DCEntity) -> bool:
    return a.ste == b.ste


def eval_pos(a: DCEntity, b: DCEntity) -> bool:
    return bool(np.linalg.norm(a.pos - b.pos) <= 0.04)


# NOTE: Reference OGBench success
# We keep eval functionality the same but have to pull it out of the scene for subgoal evaluation

# def _compute_successes(self):
#     """Compute object successes."""
#     cube_successes = []
#     for i in range(self._num_cubes):
#         obj_pos = self._data.joint(f"object_joint_{i}").qpos[:3]
#         tar_pos = self._data.mocap_pos[self._cube_target_mocap_ids[i]]
#         if np.linalg.norm(obj_pos - tar_pos) <= 0.04:
#             cube_successes.append(True)
#         else:
#             cube_successes.append(False)
#     button_successes = [
#         (self._cur_button_states[i] == self._target_button_states[i])
#         for i in range(self._num_buttons)
#     ]
#     drawer_success = (
#         np.abs(self._data.joint("drawer_slide").qpos[0] - self._target_drawer_pos)
#         <= 0.04
#     )
#     window_success = (
#         np.abs(self._data.joint("window_slide").qpos[0] - self._target_window_pos)
#         <= 0.04
#     )

#     return cube_successes, button_successes, drawer_success, window_success
