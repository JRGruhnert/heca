import math
import warnings
from typing import Any

import numpy as np
from dataclasses import dataclass, field

from heca.misc.base import Configurable
from heca.data.data import DCEntity, DCScene
from heca.utils.quaternion import Quaternion


class Entity(Configurable):
    FEATURE_DIM: int = 33  # 16 (logits) + 13 (pose) + 2*2 (max extra for revolute)
    MAX_STATE_DIM: int = 16
    BASE_LOGSTD = -10.0
    LOGIT_CONFIDENCE = 10.0
    POS_DIM: int = 3
    ROT_DIM: int = 3
    TYPE_ID: int = -1  # overridden by subclasses

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        threshold: float
        fit_rotation: bool = True
        directional_containment: bool = False
        n_states: int = 1
        question: str = ""
        answers: list[str] = field(default_factory=list)
        max_fit_components: int = 10
        # Pose gate: max allowed sigma (Mahalanobis) deviation of a value from
        # the best mixture component. ``None`` derives it from ``threshold``
        # via ``sqrt(-2*ln(threshold))``, which reproduces the legacy single
        # threshold on the pose part exactly (0.9 -> 0.46 sigma, 0.4 -> 1.35).
        z_threshold: float | None = None

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def rot_dim(self) -> int:
        return Entity.ROT_DIM if self.cfg.fit_rotation else 0

    def model_value(self, value: np.ndarray) -> np.ndarray:
        """Map a full value [pos | aa | extra | ste] to the model input.

        When ``fit_rotation`` is False the axis-angle columns are dropped, so
        the StepMix model only ever sees [pos | extra | ste].
        """
        value = np.asarray(value)
        if self.cfg.fit_rotation:
            return value
        return np.concatenate(
            [value[..., : self.POS_DIM], value[..., self.POS_DIM + self.ROT_DIM :]],
            axis=-1,
        )

    def model_to_value(self, value: np.ndarray) -> np.ndarray:
        """Map model output [pos | extra | ste] back to a full value.

        Reinserts a zero (identity) axis-angle rotation so the result always has
        the canonical [pos | aa | extra | ste] layout expected downstream.
        """
        value = np.asarray(value)
        if self.cfg.fit_rotation:
            return value
        zeros = np.zeros(value.shape[:-1] + (self.ROT_DIM,), dtype=value.dtype)
        return np.concatenate(
            [value[..., : self.POS_DIM], zeros, value[..., self.POS_DIM :]],
            axis=-1,
        )

    @property
    def measurement(self) -> dict:
        raise NotImplementedError

    def normalize_position(self, pos, obs) -> np.ndarray:
        return (pos - obs["meta_xyz_center"]) * obs["meta_xyz_scaler"]

    def unnormalize_position(self, pos, center, scaler) -> np.ndarray:
        """Inverse of ``normalize_position`` (raw = norm / scaler + center)."""
        return np.asarray(pos, dtype=np.float32) / np.asarray(
            scaler, dtype=np.float32
        ) + np.asarray(center, dtype=np.float32)

    def common_pose_part(self, label: str, obs: dict) -> np.ndarray:
        pos = obs[f"heca_{label}_pos"]
        rot = obs[f"heca_{label}_rot"]
        pos = self.normalize_position(pos, obs)
        # heca_*_rot is already (w, x, y, z); keep it before log-mapping.
        rot = np.array(rot, dtype=np.float32)
        quat = Quaternion.normalize(rot)
        aa = Quaternion.log_map(quat)
        return np.concatenate((pos, aa))

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def value_from_gt(self, label: str, obs: dict) -> DCEntity:
        pose = self.common_pose_part(label, obs)
        extra = self.extra_part(label, obs)
        ste = obs[f"heca_{label}_ste"]
        return self.dc_from_parsed(pose, extra, ste)

    def dc_from_parsed(
        self, pose: np.ndarray, extra: np.ndarray, ste: np.ndarray
    ) -> DCEntity:
        value = np.concatenate(
            (
                np.asarray(pose).ravel(),
                np.asarray(extra).ravel(),
                np.asarray(ste).ravel(),
            )
        )
        feature = self.gnn_format(value)
        return DCEntity(value=value, feature=feature)

    def env_state_value(self, label: str, x: DCScene) -> dict[str, Any]:
        raise NotImplementedError

    def value_from_image(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def gnn_format(self, value: np.ndarray) -> np.ndarray:
        M = Entity.MAX_STATE_DIM
        D = len(value) - 7  # pos(3) + aa(3) + ste(1) = 7 base, extra is rest

        feat = np.zeros(Entity.FEATURE_DIM, dtype=np.float32)

        state_id = value[6 + D].astype(int)
        feat[: self.cfg.n_states] = -self.LOGIT_CONFIDENCE
        if state_id < self.cfg.n_states:
            feat[state_id] = self.LOGIT_CONFIDENCE

        feat[M : M + 3] = value[0:3]
        feat[M + 3 : M + 6] = self.BASE_LOGSTD

        quat = Quaternion.exp(value[3:6])
        feat[M + 6 : M + 10] = Quaternion.normalize(quat)
        feat[M + 10 : M + 13] = self.BASE_LOGSTD

        if D > 0:
            feat[M + 13 : M + 13 + D] = value[6 : 6 + D]
            feat[M + 13 + D : M + 13 + 2 * D] = self.BASE_LOGSTD

        return feat

    def make_agent_key(self, label: str, obs: Any, start: int, end: int) -> str:
        raise NotImplementedError

    def secure_mix_parameters(self, p: dict, eps: float = 1e-15) -> dict:
        pis = p["measurement"]["state"]["pis"]
        # Pad to at least cfg.n_states; if the fitted categorical has *more*
        # outcomes than configured, keep them all (instead of crashing) and warn,
        # because the GNN feature encoders can only represent cfg.n_states.
        n_outcomes = max(self.cfg.n_states, pis.shape[1])
        padded = np.full((pis.shape[0], n_outcomes), eps, dtype=np.float32)
        padded[:, : pis.shape[1]] = pis
        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        p["measurement"]["state"]["pis"] = padded
        if pis.shape[1] > self.cfg.n_states:
            warnings.warn(
                f"Fitted categorical has {pis.shape[1]} outcomes but "
                f"cfg.n_states={self.cfg.n_states}. Increase n_states in the "
                "entity config so the GNN features do not silently drop states.",
                stacklevel=2,
            )
        return p

    def score_single(
        self, sample: np.ndarray, up: dict, eps: float = 1e-15
    ) -> tuple[float, bool]:
        """Score a single sample under a StepMix model. Returns ``(score, valid)``.

        ``score`` (in ``[0, 1]``) is the sample's likelihood relative to the
        model's peak (1.0 = exactly at the most probable point); it is kept as
        a diagnostic/ranking value.

        ``valid`` is the conjunction of two gates:

        * pose gate: the Mahalanobis distance ``z`` (in sigma units) of the
          sample to the best (highest-posterior) component must be
          ``<= z_threshold``. If ``z_threshold`` is ``None`` it defaults to
          ``sqrt(-2*ln(threshold))``, reproducing the legacy single-threshold
          behavior on the pose part exactly (0.9 -> 0.46 sigma, 0.4 -> 1.35).
        * state gate: a plain equality check — the observed state must equal
          the best component's most likely state. No tuning knob: if the
          state does not match, the value is not in the condition.

        Because the pose gate is scale-free (sigma deviation) and the state
        gate is a plain categorical match, they are comparable across entity
        types (e.g. static vs free-moving).
        """
        sample = self.model_value(sample)
        p = self.secure_mix_parameters(up)
        pose = sample[:-1]
        state = int(sample[-1])
        weights = p["weights"]
        means = p["measurement"]["pose"]["means"]
        vars_ = p["measurement"]["pose"]["covariances"]
        pis = p["measurement"]["state"]["pis"]

        def loglik(pose_x: np.ndarray, state_x: int) -> float:
            best = -np.inf
            for k in range(len(weights)):
                var = np.maximum(vars_[k], eps)
                log_gauss = -0.5 * np.sum(
                    np.log(2 * np.pi * var) + (pose_x - means[k]) ** 2 / var
                )
                state_prob = pis[k][state_x] if state_x < len(pis[k]) else eps
                log_cat = np.log(np.clip(state_prob, eps, 1.0))
                s = np.log(weights[k]) + log_gauss + log_cat
                if s > best:
                    best = s
            return best

        loglik_x = loglik(pose, state)
        loglik_max = max(
            loglik(means[k], int(np.argmax(pis[k]))) for k in range(len(weights))
        )
        score = float(np.clip(np.exp(loglik_x - loglik_max), 0.0, 1.0))

        # --- pose gate: sigma deviation to the best component ---
        best_k, z = self._best_component(pose, p, eps=eps)
        valid_pose = z <= self._effective_z_threshold()

        # --- state gate: hard equality with the best component's most likely
        # state. Not tunable: a mismatching state means "not in the condition".
        mode_state = int(np.argmax(pis[best_k]))
        valid_state = state == mode_state

        return score, bool(valid_pose and valid_state)

    def _effective_z_threshold(self) -> float:
        if self.cfg.z_threshold is not None:
            return float(self.cfg.z_threshold)
        if self.cfg.threshold <= 0.0:
            return float("inf")
        return float(math.sqrt(-2.0 * math.log(self.cfg.threshold)))

    def _best_component(
        self, pose: np.ndarray, p: dict, eps: float = 1e-15
    ) -> tuple[int, float]:
        """Index of the highest-posterior component for ``pose`` (weight *
        Gaussian density, ignoring the state) and the pose's Mahalanobis
        distance (sigma deviation) to that component."""
        weights = p["weights"]
        means = p["measurement"]["pose"]["means"]
        vars_ = p["measurement"]["pose"]["covariances"]
        best_k, best_post = -1, -np.inf
        for k in range(len(weights)):
            var = np.maximum(vars_[k], eps)
            post = np.log(weights[k]) - 0.5 * np.sum(
                np.log(2 * np.pi * var) + (pose - means[k]) ** 2 / var
            )
            if post > best_post:
                best_k, best_post = k, post
        z = float(
            np.sqrt(np.sum((pose - means[best_k]) ** 2 / np.maximum(vars_[best_k], eps)))
        )
        return best_k, z

    def sigma_deviation(self, sample: np.ndarray, up: dict, eps: float = 1e-15) -> float:
        """Mahalanobis distance (in sigma units) from ``sample`` to the best
        (highest-posterior) component of the model. Useful for tuning
        ``z_threshold``."""
        sample = self.model_value(sample)
        p = self.secure_mix_parameters(up)
        _, z = self._best_component(sample[:-1], p, eps=eps)
        return z

    @staticmethod
    def _gaussian_overlap(
        mu1: np.ndarray,
        var1: np.ndarray,
        mu2: np.ndarray,
        var2: np.ndarray,
        directional: bool = False,
        eps: float = 1e-15,
    ) -> np.ndarray:
        """Per-dim Gaussian overlap in (0, 1].

        Symmetric (Bhattacharyya coefficient) by default; when ``directional`` is
        set the width term instead penalizes a source (``2``) that is wider than
        the target (``1``), i.e. "does the source fit inside the target". The
        distance term always uses ``var1 + var2``.
        """
        v1 = np.maximum(var1, eps)
        v2 = np.maximum(var2, eps)
        var_sum = v1 + v2
        if directional:
            # source (2) must fit inside target (1); ==1 when v1 >= v2
            width = np.minimum(1.0, np.sqrt(2.0 * v1 / var_sum))
        else:
            width = np.sqrt(2.0 * np.sqrt(v1 * v2) / var_sum)  # =1 when v1 == v2
        distance = np.exp(-0.25 * (mu1 - mu2) ** 2 / var_sum)  # =1 when means equal
        return width * distance

    def containment_score(self, up1: dict, up2: dict):
        """How much of ``up2``'s mass falls inside ``up1``'s distribution. [0, 1].

        Normalized by the self-overlap of ``up1``, so a distribution always
        contains itself with score 1.0 and the value does not depend on the
        number of mixture components or the weight split. (Previously a
        multi-component model scored only ~sum(w_i^2) against itself — 0.5 for
        two equal one-hot components — which could never reach the 0.9 static
        threshold.)
        """
        p1 = self.secure_mix_parameters(up1)
        p2 = self.secure_mix_parameters(up2)
        overlap = self._pair_overlap(p1, p2)
        self_overlap = self._pair_overlap(p1, p1)
        if self_overlap <= 0.0:
            return 0.0
        return float(np.clip(overlap / self_overlap, 0.0, 1.0))

    def _pair_overlap(self, p1: dict, p2: dict) -> float:
        """Expected overlap between two mixtures (un-normalized, in [0, 1])."""
        score = 0.0
        for i in range(len(p1["weights"])):
            w_i = p1["weights"][i]
            for j in range(len(p2["weights"])):
                w_j = p2["weights"][j]
                mu1 = p1["measurement"]["pose"]["means"][i]
                var1 = p1["measurement"]["pose"]["covariances"][i]
                mu2 = p2["measurement"]["pose"]["means"][j]
                var2 = p2["measurement"]["pose"]["covariances"][j]
                cat1 = p1["measurement"]["state"]["pis"][i]
                cat2 = p2["measurement"]["state"]["pis"][j]
                gauss_rel = np.prod(
                    self._gaussian_overlap(
                        mu1,
                        var1,
                        mu2,
                        var2,
                        directional=self.cfg.directional_containment,
                    )
                )
                # Align categorical distributions to the union of observed states.
                n = max(len(cat1), len(cat2))
                if len(cat1) < n:
                    cat1 = np.pad(cat1, (0, n - len(cat1)))
                if len(cat2) < n:
                    cat2 = np.pad(cat2, (0, n - len(cat2)))
                overlap_cat = np.sum(cat1 * cat2)
                peak_target = np.max(cat1)
                cat_score = overlap_cat / peak_target if peak_target > 0 else 0.0
                score += w_i * w_j * gauss_rel * cat_score
        return float(score)

    def best_sample(self, up1: dict, up2: dict, eps: float = 1e-15):
        p1 = self.secure_mix_parameters(up1)
        p2 = self.secure_mix_parameters(up2)

        weights1 = p1["weights"]
        weights2 = p2["weights"]
        means1 = p1["measurement"]["pose"]["means"]
        means2 = p2["measurement"]["pose"]["means"]
        vars1 = p1["measurement"]["pose"]["covariances"]
        vars2 = p2["measurement"]["pose"]["covariances"]
        state1 = p1["measurement"]["state"]["pis"]
        state2 = p2["measurement"]["state"]["pis"]

        # Align categorical distributions to the union of observed states
        K1, K2 = len(weights1), len(weights2)
        results = []
        for i in range(K1):
            for j in range(K2):
                # Gaussian diag part
                precision = 1.0 / vars1[i] + 1.0 / vars2[j]
                var = 1.0 / precision
                mean = var * (means1[i] / vars1[i] + means2[j] / vars2[j])
                diff = means1[i] - means2[j]
                log_norm = -0.5 * (
                    np.sum(np.log(2 * np.pi * (vars1[i] + vars2[j])))
                    + np.sum(diff**2 / (vars1[i] + vars2[j]))
                )
                # Categorical part — padded to the union of observed states
                n = max(len(state1[i]), len(state2[j]))
                c1 = np.pad(state1[i], (0, n - len(state1[i]))) if len(state1[i]) < n else state1[i]
                c2 = np.pad(state2[j], (0, n - len(state2[j]))) if len(state2[j]) < n else state2[j]
                cat_prod = c1 * c2  # element-wise over aligned states
                state = int(np.argmax(cat_prod))
                if cat_prod[state] <= 0.0:
                    # Disjoint categorical supports: pick the state with the
                    # highest combined mass instead of an arbitrary argmax(0).
                    state = int(np.argmax(c1 + c2))
                log_cat = np.log(np.clip(cat_prod[state], eps, None))
                score = np.log(weights1[i]) + np.log(weights2[j]) + log_norm + log_cat
                results.append({"score": score, "pose": mean, "state": state})

        results.sort(key=lambda r: r["score"], reverse=True)
        pose = results[0]["pose"]
        state = results[0]["state"]
        assert isinstance(pose, np.ndarray)
        return self.model_to_value(np.concatenate([pose, [state]]))

    def comp_feature(self, up: dict, eps: float = 1e-8) -> np.ndarray:
        """
        NOTE: ASSUMES MODELS USE DIAG MODE
        Returns:
            np.ndarray of shape (N, FEATURE_DIM) with layout:
                - [0:MAX_STATE_DIM]             = state scores on the same
                  [-LOGIT_CONFIDENCE, +LOGIT_CONFIDENCE] scale as gnn_format
                  (2*p - 1) * LOGIT_CONFIDENCE; impossible states stay at -10
                - [M:M+3]                       = μ_pos
                - [M+3:M+6]                     = log(σ_pos)
                - [M+6:M+10]                    = quaternion [w, x, y, z]
                - [M+10:M+13]                   = log(σ_rot, tangent space)
                - [M+13:M+13+D]                 = μ_extra
                - [M+13+D:M+13+2D]              = log(σ_extra)
            where M = MAX_STATE_DIM, D = n_columns - 6
        """

        p = self.secure_mix_parameters(up)
        means = p["measurement"]["pose"]["means"]  # (N, n_columns)
        covariances = p["measurement"]["pose"]["covariances"]  # (N, n_columns)
        pis = p["measurement"]["state"]["pis"]  # (N, K)
        N = len(p["weights"])
        M = Entity.MAX_STATE_DIM
        base = Entity.POS_DIM + (Entity.ROT_DIM if self.cfg.fit_rotation else 0)
        D = means.shape[1] - base  # extra continuous dims beyond pos (+rot)

        feat = np.zeros((N, Entity.FEATURE_DIM), dtype=np.float32)

        # State scores — same convention and scale as gnn_format.
        feat[:, : self.cfg.n_states] = -self.LOGIT_CONFIDENCE
        n_logit = min(pis.shape[1], self.cfg.n_states)
        feat[:, :n_logit] = self.LOGIT_CONFIDENCE * (2.0 * pis[:, :n_logit] - 1.0)

        # Pose: position mean + logstd
        feat[:, M : M + 3] = means[:, 0:3]
        feat[:, M + 3 : M + 6] = 0.5 * np.log(covariances[:, 0:3] + eps)

        # Pose: axis-angle -> quaternion + rotation logstd (3 dof in tangent space)
        if self.cfg.fit_rotation:
            quat = Quaternion.exp(means[:, 3:6])
            feat[:, M + 6 : M + 10] = Quaternion.normalize(quat)
            feat[:, M + 10 : M + 13] = 0.5 * np.log(covariances[:, 3:6] + eps)
        else:
            feat[:, M + 6 : M + 10] = Quaternion.identity()
            feat[:, M + 10 : M + 13] = self.BASE_LOGSTD

        # Extra continuous dims: mean + logstd
        if D > 0:
            feat[:, M + 13 : M + 13 + D] = means[:, base:]
            feat[:, M + 13 + D : M + 13 + 2 * D] = 0.5 * np.log(
                covariances[:, base:] + eps
            )

        return feat
