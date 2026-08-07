import numpy as np
from dataclasses import dataclass


import torch

from heca.misc.base import Configurable
from heca.data.data import DCEntity
from heca.utils.quaternion import Quaternion

STATE_LOGIT_BASELINE = -10.0


class Entity(Configurable):
    input_feat_dim: int = 56

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        states: list[str] = []
        question: str = ""
        answers: list[str] = []

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def n_states(self) -> int:
        return len(self.cfg.states)

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.states), dtype=torch.float32)

    def make_idx(self, label: str) -> int:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.states, "Label must be in state values."
        return self.cfg.states.index(label)

    def one_hot_from_idx(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        assert 0 <= idx < len(self.cfg.states), "Index out of bounds."
        one_hot = torch.zeros(len(self.cfg.states), dtype=torch.float32)
        one_hot[idx] = 1.0
        return one_hot

    @classmethod
    def one_hot_from_idx_dc(cls, idx: int, n_states: int) -> np.ndarray:
        assert 0 <= idx < n_states, "Index out of bounds."
        one_hot = np.zeros(n_states)
        one_hot[idx] = 1.0
        return one_hot

    def value_from_gt(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def value_from_image(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def gnn_format(self, value: np.ndarray):
        raise NotImplementedError

    @classmethod
    def apply_artificial_uncertainty(
        cls,
        entity_features: np.ndarray,
        pos_noise_std=0.1,
        rot_noise_std=0.1,
        state_noise_std=0.5,
        base_eps=1e-5,
        rng=None,
    ):
        if rng is None:
            rng = np.random.default_rng()

        # Copy to avoid modifying original
        noisy = entity_features.copy()
        N = entity_features.shape[0]
        K = entity_features.shape[1] - 13

        # Unpack
        mu_pos = entity_features[:, 0:3]
        logstd_pos = entity_features[:, 3:6]
        q = entity_features[:, 6:10]
        logstd_rot = entity_features[:, 10:13]
        logits_state = entity_features[:, 13:]

        # --- 1. Position Noise ---
        # Sample noise in world space
        delta_pos = rng.normal(0, pos_noise_std, size=(N, 3))
        # Update mean
        noisy[:, 0:3] = mu_pos + delta_pos
        # Update variance: combine existing variance (from logstd) + injected noise variance
        existing_var_pos = np.exp(2 * logstd_pos)
        new_var_pos = existing_var_pos + pos_noise_std**2
        # Clamp variance to ensure we don't go below a tiny epsilon (prevents -inf later)
        new_var_pos = np.maximum(new_var_pos, base_eps**2)
        # Store new log-std
        noisy[:, 3:6] = 0.5 * np.log(new_var_pos)

        # --- 2. Rotation Noise ---
        # Sample 3D tangent-space noise (axis-angle)
        delta_rot = rng.normal(0, rot_noise_std, size=(N, 3))
        # Convert noise to quaternion: q_noise = Exp(delta_rot)
        q_noise = Quaternion.exp(delta_rot)
        # Update mean: q_new = q_noise ⊗ q_clean (order matters: apply noise locally)
        noisy_q = Quaternion.mul(q_noise, q)  # [N, 4]
        # Normalize (just in case)
        noisy_q = noisy_q / np.linalg.norm(noisy_q, axis=-1, keepdims=True)
        noisy[:, 6:10] = noisy_q

        # Update rotational covariance
        existing_var_rot = np.exp(2 * logstd_rot)
        new_var_rot = existing_var_rot + rot_noise_std**2
        new_var_rot = np.maximum(new_var_rot, base_eps**2)
        noisy[:, 10:13] = 0.5 * np.log(new_var_rot)

        # --- 3. State Noise ---
        # Add Gaussian noise directly to the logits
        delta_logits = rng.normal(0, state_noise_std, size=(N, K))
        noisy_logits = logits_state + delta_logits
        # No need to update a "log-std" for state here, as we just perturb the logits.
        # The GNN will interpret the softened logits as a less confident measurement.
        noisy[:, 13:] = noisy_logits

        return noisy
