from typing import Callable

import numpy as np

from dataclasses import dataclass
from enum import Enum
from functools import total_ordering

import torch

from heca.misc.base import Configurable
from heca.misc.data import DCEntity
from heca.utils.quaternion import Quaternion

STATE_LOGIT_BASELINE = -10.0


class Mobility(Enum):
    FREE = "free"  # Can move freely in the scene
    STATIC = "static"  # Has a fixed position and rotation in the scene
    ARTICULATED = "articulated"  # Has a fixed position but can have a variable rotation in the scene


@total_ordering
class Entity(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        label: str
        scene: str
        states: set[str]
        question: str
        answers: set[str]
        mobility: Mobility
        eval_func: Callable[[np.ndarray, np.ndarray], bool] = lambda a, b: False
        max_states: int = 19  # 13 (pos+rot) + 19 = 32 (dim of gnn features)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    def state_count(self) -> int:
        return len(self.cfg.states)

    def evaluate(self, a: DCEntity, b: DCEntity) -> bool:
        return self.cfg.eval_func(a.value, b.value)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.cfg.label == other.cfg.label

    def __hash__(self) -> int:
        return hash(self.cfg.label)

    def __lt__(self, other: "Entity") -> bool:
        if not isinstance(other, Entity):
            return NotImplemented
        return self.cfg.label < other.cfg.label

    def make_zeros(self) -> torch.Tensor:
        return torch.zeros(len(self.cfg.states), dtype=torch.float32)

    def make_zeros_dc(self) -> np.ndarray:
        return np.zeros(len(self.cfg.states))

    def make_one_hot(self, label: str) -> torch.Tensor:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.states, "Label must be in state values."
        one_hot = self.make_zeros()
        index = list(self.cfg.states).index(label)
        one_hot[index] = 1.0
        return one_hot

    def make_idx(self, label: str) -> int:
        assert label is not None, "Label cannot be None."
        assert label in self.cfg.states, "Label must be in state values."
        return list(self.cfg.states).index(label)

    def one_hot_from_idx(self, idx: int) -> torch.Tensor:
        idx = int(idx)
        assert 0 <= idx < len(self.cfg.states), "Index out of bounds."
        one_hot = self.make_zeros()
        one_hot[idx] = 1.0
        return one_hot

    def one_hot_from_idx_dc(self, idx: int) -> np.ndarray:
        idx = int(idx)
        assert 0 <= idx < len(self.cfg.states), "Index out of bounds."
        one_hot = self.make_zeros_dc()
        one_hot[idx] = 1.0
        return one_hot

    @classmethod
    def to_value(
        cls, pos: np.ndarray, rot: np.ndarray, ste: np.ndarray, soh: np.ndarray
    ) -> DCEntity:
        value = np.concatenate((pos, rot, ste[:, None]), axis=1)
        feature = cls.gnn_format(value, len(soh))
        return DCEntity(value=value, feature=feature)

    @classmethod
    def gnn_format(
        cls, raw: np.ndarray, n_states: int, logit_confidence=10.0, base_logstd=-10.0
    ):
        """
        Convert raw stepmix format to GNN node format.

        Args:
            raw_entities: [N, 8] where columns are:
                        [pos_x, pos_y, pos_z, qw, qx, qy, qz, state_scalar]
            K: Number of state categories.
            logit_confidence: Value to assign to the true class logit.
            base_logstd: Initial log-std for deterministic entities (e.g., -10).

        Returns:
            gnn_nodes: [N, 13 + K] with structure:
                    [μ_pos(3), logσ_pos(3), q(4), logσ_rot(3), logits_state(K)]
        """
        # Initialize with zeros
        node = np.zeros((32), dtype=np.float32)
        node[0:3] = raw[0:3]
        node[3:6] = base_logstd
        node[6:10] = raw[3:7]
        norms = np.linalg.norm(node[6:10], axis=-1, keepdims=True)
        node[6:10] = node[6:10] / norms
        node[10:13] = base_logstd
        state_ids = raw[7].astype(int)  # [N]
        node[13 : 13 + n_states] = -logit_confidence
        node[13 + state_ids] = logit_confidence

        return node

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
        """
        Inject Gaussian noise into Real/Entity nodes and update their stored uncertainty.

        Args:
            entity_features: [N, 13+K] Clean or base entity features.
                            Structure: [μ_pos(3), logσ_pos(3), q(4), logσ_rot(3), logits_state(K)]
            pos_noise_std: Standard deviation of position noise (in world units).
            rot_noise_std: Standard deviation of rotation noise (in radians, in tangent space).
            state_noise_std: Standard deviation of noise added to logits.
            base_eps: Minimum standard deviation to keep for numerical stability (if noise is 0).
            rng: np.random.Generator for reproducibility.

        Returns:
            noisy_entities: [N, 13+K] Modified entity features with updated means and log-stds.
        """
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

    def gnn_format2(
        self,
        raw: np.ndarray,
        logit_confidence=10.0,
        base_logstd=-10.0,
        pos_logstd: np.ndarray | None = None,
        rot_logstd: np.ndarray | None = None,
        state_logits: np.ndarray | None = None,
    ):
        """
        Convert raw stepmix format to GNN node format.

        Args:
            raw: [8] where columns are [pos_x, pos_y, pos_z, qw, qx, qy, qz, state_scalar]
            logit_confidence: Value to assign to the true class logit (when not overridden).
            base_logstd: Initial log-std for deterministic entities (when not overridden).
            pos_logstd: [3] override for position log-std (from StepMix model).
            rot_logstd: [3] override for rotation log-std (from StepMix model).
            state_logits: [K] override for state logits (from StepMix model).

        Returns:
            gnn_node: [13 + K] with structure:
                    [μ_pos(3), logσ_pos(3), q(4), logσ_rot(3), logits_state(K)]
        """
        K = self.state_count()
        node = np.zeros((13 + K), dtype=np.float32)

        # 1. Position mean (raw pos)
        node[0:3] = raw[0:3]

        # 2. Position log-std (from model if available, else deterministic)
        if pos_logstd is not None:
            node[3:6] = pos_logstd
        else:
            node[3:6] = base_logstd

        # 3. Quaternion mean (raw quat)
        node[6:10] = raw[3:7]
        norms = np.linalg.norm(node[6:10], axis=-1, keepdims=True)
        node[6:10] = node[6:10] / norms

        # 4. Rotation log-std (from model if available, else deterministic)
        if rot_logstd is not None:
            node[10:13] = rot_logstd
        else:
            node[10:13] = base_logstd

        # 5. State logits (from model if available, else one-hot-style high confidence)
        if state_logits is not None:
            node[13:] = state_logits
        else:
            state_ids = int(raw[7])
            node[13:] = -logit_confidence
            node[13 + state_ids] = logit_confidence

        return node
