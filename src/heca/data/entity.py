from typing import Any

import numpy as np
from dataclasses import dataclass, field

from heca.misc.base import Configurable
from heca.data.data import DCEntity
from heca.utils.quaternion import Quaternion


class Entity(Configurable):
    FEATURE_DIM: int = 35  # 16 (logits) + 13 (pose) + 6 (max extra: prismatic)
    MAX_STATE_DIM: int = 16
    BASE_LOGSTD = -10.0
    LOGIT_CONFIDENCE = 10.0
    TYPE_ID: int = -1  # overridden by subclasses

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        threshold: float
        states: list[str] = field(default_factory=list)
        question: str = ""
        answers: list[str] = field(default_factory=list)

    def __init__(self, cfg: Config):
        self.cfg = cfg

    @property
    def n_states(self) -> int:
        return len(self.cfg.states)

    @property
    def measurement(self) -> dict:
        raise NotImplementedError

    def normalize_position(self, pos, obs) -> np.ndarray:
        return (pos - obs["meta_xyz_center"]) * obs["meta_xyz_scaler"]

    def common_pose_part(self, label: str, obs: dict) -> np.ndarray:
        pos = obs[f"heca_{label}_pos"]
        rot = obs[f"heca_{label}_rot"]
        rot = np.array([rot[1], rot[2], rot[3], rot[0]], dtype=np.float32)
        return np.concatenate((pos, rot))

    def extra_part(self, label: str, obs: dict) -> np.ndarray:
        raise NotImplementedError

    def value_from_gt(self, label: str, obs: dict) -> DCEntity:
        pose = self.common_pose_part(label, obs)
        extra = self.extra_part(label, obs)
        ste = obs[f"heca_{label}_ste"]
        value = np.concatenate((pose, extra, ste))
        feature = self.gnn_format(value)
        return DCEntity(value=value, feature=feature)

    def value_from_image(self, obs: dict) -> DCEntity:
        raise NotImplementedError

    def gnn_format(self, value: np.ndarray) -> np.ndarray:
        """Encode a ground-truth value into the GNN feature vector.

        value layout: [pos(3), quat(4), extra(D), state_id(1)]
        D = len(value) - 8  (extra continuous dims beyond the 7 pose dims)
        """
        M = Entity.MAX_STATE_DIM
        D = len(value) - 8  # extra dims

        feat = np.zeros(Entity.FEATURE_DIM, dtype=np.float32)

        # State logits (GT: one-hot with confidence markers)
        state_id = value[7 + D].astype(int)
        feat[: self.n_states] = -self.LOGIT_CONFIDENCE
        feat[state_id] = self.LOGIT_CONFIDENCE

        # Pose: position mean + assumed std
        feat[M : M + 3] = value[0:3]
        feat[M + 3 : M + 6] = self.BASE_LOGSTD

        # Pose: quaternion + assumed rotation std
        feat[M + 6 : M + 10] = Quaternion.normalize(value[3:7])
        feat[M + 10 : M + 13] = self.BASE_LOGSTD

        # Extra continuous dims: mean + assumed std
        if D > 0:
            feat[M + 13 : M + 13 + D] = value[7 : 7 + D]
            feat[M + 13 + D : M + 13 + 2 * D] = self.BASE_LOGSTD

        return feat

    def make_agent_key(
        self, label: str, obs: Any, start: int, end: int
    ) -> str:
        raise NotImplementedError

    def secure_mix_parameters(self, p: dict, eps: float = 1e-15) -> dict:
        pis = p["measurement"]["state"]["pis"]
        padded = np.full((pis.shape[0], self.n_states), eps, dtype=np.float32)
        padded[:, : pis.shape[1]] = pis
        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        p["measurement"]["state"]["pis"] = padded
        return p

    def score_single(
        self, sample: np.ndarray, up: dict, eps: float = 1e-15
    ) -> tuple[float, bool]:
        """Score a single sample under a StepMix model. Returns [0,1]."""
        p = self.secure_mix_parameters(up)
        pose = sample[:7]
        state = int(sample[-1])
        best_logprob = -np.inf
        for k in range(len(p["weights"])):
            mu = p["measurement"]["pose"]["means"][k]
            var = p["measurement"]["pose"]["covariances"][k]
            pis = p["measurement"]["state"]["pis"][k]

            # Gaussian
            log_gauss = -0.5 * np.sum(np.log(2 * np.pi * var) + (pose - mu) ** 2 / var)

            # Categorical
            state_prob = pis[state] if state < len(pis) else eps

            log_cat = np.log(np.clip(state_prob, eps, 1))
            score = np.log(p["weights"][k]) + log_gauss + log_cat

            if score > best_logprob:
                best_logprob = score

        score = np.exp(best_logprob)
        valid = score >= self.cfg.threshold
        return score, valid

    def containment_score(self, up1: dict, up2: dict):
        """How much of others mass falls inside selfs distribution."""
        p1 = self.secure_mix_parameters(up1)
        p2 = self.secure_mix_parameters(up2)

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
                diff = mu1 - mu2
                pos_score = np.exp(-0.5 * np.sum(diff**2 / var1))
                sigma_target = np.sqrt(var1)
                sigma_source = np.sqrt(var2)
                per_dim_penalty = np.minimum(1.0, sigma_target / sigma_source)
                width_penalty = np.prod(per_dim_penalty)
                gauss_rel = width_penalty * pos_score
                overlap_cat = np.sum(cat1 * cat2)
                peak_target = np.max(cat1)
                cat_score = overlap_cat / peak_target if peak_target > 0 else 0.0
                score += w_i * w_j * gauss_rel * cat_score
        return score  # [0, 1]

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
                # Categorical part — padded to same target space
                cat_prod = state1[i] * state2[j]  # element-wise over aligned states
                state = int(np.argmax(cat_prod))
                log_cat = np.log(np.clip(cat_prod[state], eps, None))
                score = np.log(weights1[i]) + np.log(weights2[j]) + log_norm + log_cat
                results.append({"score": score, "pose": mean, "state": state})

        results.sort(key=lambda r: r["score"], reverse=True)
        pose = results[0]["pose"]
        state = results[0]["state"]
        assert isinstance(pose, np.ndarray)
        return np.concatenate([pose, [state]])

    def comp_feature(self, up: dict, eps: float = 1e-8) -> np.ndarray:
        """
        NOTE: ASSUMES MODELS USE DIAG MODE
        Returns:
            np.ndarray of shape (N, FEATURE_DIM) with layout:
                - [0:MAX_STATE_DIM]             = logits (state logits, unnormalized)
                - [M:M+3]                       = μ_pos
                - [M+3:M+6]                     = log(σ_pos)
                - [M+6:M+10]                    = quaternion [w, x, y, z]
                - [M+10:M+13]                   = log(σ_rot, tangent space)
                - [M+13:M+13+D]                 = μ_extra
                - [M+13+D:M+13+2D]              = log(σ_extra)
            where M = MAX_STATE_DIM, D = n_columns - 7
        """

        p = self.secure_mix_parameters(up)
        means = p["measurement"]["pose"]["means"]  # (N, n_columns)
        covariances = p["measurement"]["pose"]["covariances"]  # (N, n_columns)
        pis = p["measurement"]["state"]["pis"]  # (N, K)
        N = len(p["weights"])
        M = Entity.MAX_STATE_DIM
        D = means.shape[1] - 7  # extra continuous dims beyond base 7 pose

        feat = np.zeros((N, Entity.FEATURE_DIM), dtype=np.float32)

        # State logits
        feat[:, : self.n_states] = np.log(pis)

        # Pose: position mean + logstd
        feat[:, M : M + 3] = means[:, 0:3]
        feat[:, M + 3 : M + 6] = 0.5 * np.log(covariances[:, 0:3] + eps)

        # Pose: quaternion + rotation logstd (3 dof in tangent space)
        feat[:, M + 6 : M + 10] = Quaternion.normalize(means[:, 3:7])
        feat[:, M + 10 : M + 13] = 0.5 * np.log(covariances[:, 3:6] + eps)

        # Extra continuous dims: mean + logstd
        if D > 0:
            feat[:, M + 13 : M + 13 + D] = means[:, 7:]
            feat[:, M + 13 + D : M + 13 + 2 * D] = (
                0.5 * np.log(covariances[:, 7:] + eps)
            )

        return feat
