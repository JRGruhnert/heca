import numpy as np
from dataclasses import dataclass

from heca.misc.base import Configurable
from heca.data.data import DCEntity
from heca.utils.quaternion import Quaternion


class Entity(Configurable):
    input_feat_dim: int = 56
    max_state_dim: int = 16

    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        threshold: float
        states: list[str] = []
        question: str = ""
        answers: list[str] = []

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
        raise NotImplementedError

    def make_agent_key(
        self, label: str, obs: dict[str, list], start: int, end: int
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
            dict: Keys are model names. Values are lists of flat feature arrays,
                one per component. Each feature array has shape (13+K,) where:
                - [0:3] = μ_pos (position means)
                - [3:6] = log(σ_pos) (log standard deviations of position)
                - [6:10] = quaternion [w, x, y, z] (rotation mean)
                - [10:13] = log(σ_rot) (log standard deviations of rotation in tangent space)
                - [13:] = logits (state logits, unnormalized)
        """

        p = self.secure_mix_parameters(up)
        weights = p["weights"]  # shape: (N,)
        means = p["measurement"]["pose"]["means"]  # shape: (N, 7)
        covariances = p["measurement"]["pose"]["covariances"]  # shape: (N, 7)
        pis = p["measurement"]["state"]["pis"]  # shape: (N, K)
        N = len(weights)

        feat = np.zeros((N, Entity.input_feat_dim), dtype=np.float32)
        feat[:, 0:3] = means[:, 0:3]  # [3]
        feat[:, 3:6] = 0.5 * np.log(covariances[:, 0:3] + eps)
        feat[:, 6:10] = Quaternion.normalize(means[:, 3:7])
        feat[:, 10:13] = 0.5 * np.log(covariances[:, 3:6] + eps)
        logits = np.log(pis)  # [K]
        feat[:, 13 : 13 + self.n_states] = logits

        return feat
