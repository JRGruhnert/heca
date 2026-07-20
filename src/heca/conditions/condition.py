from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix
from heca.misc import logger
from heca.misc.entity import Entity
from heca.utils.quaternion import Quaternion


class Condition:
    def __init__(
        self,
        label: str,
        data: dict[str, np.ndarray],
        max_components: int,
        n_samples: int,
        threshold: float,
    ):
        self.threshold = threshold
        self._data_raw = data
        self._max_components = max_components
        self._n_samples = n_samples
        self.label = label
        self.measurement = {
            "pose": {
                "model": "gaussian_diag",
                "n_columns": 7,
            },
            "state": {
                "model": "categorical",
                "n_columns": 1,
            },
        }

        self._model, self._samples, self._bics = self._fit_model()

    def comp_features(
        self, entities: dict[str, Entity], eps: float = 1e-8
    ) -> dict[str, np.ndarray]:
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

        result: dict[str, np.ndarray] = {}

        for key in self.models.keys():
            # Extract raw parameters from the fitted StepMix model
            n_states = entities[key].n_states
            params = self.secure_mix_parameters(key, n_states - 1)
            weights = params["weights"]  # shape: (N,)
            means = params["measurement"]["pose"]["means"]  # shape: (N, 7)
            covariances = params["measurement"]["pose"]["covariances"]  # shape: (N, 7)
            pis = params["measurement"]["state"]["pis"]  # shape: (K, total_outcomes)
            N = len(weights)

            feat = np.zeros((N, Entity.input_feat_dim), dtype=np.float32)
            feat[:, 0:3] = means[:, 0:3]  # [3]
            feat[:, 3:6] = 0.5 * np.log(covariances[:, 0:3] + eps)
            feat[:, 6:10] = Quaternion.normalize(means[:, 3:7])
            feat[:, 10:13] = 0.5 * np.log(covariances[:, 3:6] + eps)
            logits = np.log(pis)  # [K]
            feat[:, 13 : 13 + n_states] = logits
            result[key] = feat
        return result

    @property
    def model_states(self) -> dict[str, set[int]]:
        states: dict[str, set[int]] = {}
        for key, values in self.data_raw.items():
            states[key] = set(values[:, -1].astype(int))
        return states

    @property
    def data_raw(self) -> dict[str, np.ndarray]:
        return self._data_raw

    @property
    def samples(self) -> dict[str, np.ndarray]:
        return self._samples

    @property
    def sample_self_scores(self) -> dict[str, float]:
        return {k: float(self.models[k].score(v)) for k, v in self.samples.items()}

    @property
    def raw_self_scores(self) -> dict[str, float]:
        return {k: float(self.models[k].score(v)) for k, v in self.data_raw.items()}

    @property
    def models(self) -> dict[str, StepMix]:
        return self._model

    @property
    def elabels(self) -> set[str]:
        return set(self.data_raw.keys())

    def _fit_model(self) -> tuple[
        dict[str, StepMix],
        dict[str, np.ndarray],
        dict[str, list[float]],
    ]:

        models: dict[str, StepMix] = {}
        samples: dict[str, np.ndarray] = {}
        bics: dict[str, list[float]] = {}

        for key, values in self.data_raw.items():
            best_model = None
            best_bic = np.inf
            bic_values: list[float] = []

            for k in range(1, self._max_components + 1):
                model = StepMix(
                    n_components=k,
                    measurement=self.measurement,  # type: ignore
                )

                model.fit(values)
                bic = model.bic(values)
                bic_values.append(bic)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model

            assert best_model is not None
            models[key] = best_model
            samples[key] = best_model.sample(self._n_samples)[0]
            bics[key] = bic_values

        return models, samples, bics

    def score(self, x: dict[str, np.ndarray]) -> dict[str, float]:
        scores: dict[str, float] = {}
        for key, model in self.models.items():
            raw = model.score(x)
            delta = raw - self.sample_self_scores[key]
            clipped = np.minimum(delta, 0)  # we just care for negative deltas
            scores[key] = np.exp(clipped)
        return scores

    def plot(self, path: Path, label: str):
        ks = range(1, self._max_components + 1)
        plt.figure(figsize=(8, 5))
        for name, bic in self._bics.items():
            plt.plot(ks, bic, marker="o", label=f"{name}")
        plt.xlabel("Number of latent classes")
        plt.ylabel("BIC")
        plt.title("Model selection")
        plt.grid(True)
        plt.legend()
        plt.savefig(
            path / f"bic_{label}_{self.label}.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def make_subgoal(
        self, other: "Condition", label: str
    ) -> dict[str, tuple[float, np.ndarray]] | None:
        values = {}
        logger.debug(f"{label}")
        for key in self.elabels.intersection(other.elabels):
            score = self.containment_score(other, key)
            value = self.best_sample(other, key)
            values[key] = (score, value)
            logger.debug(f"{key}: score={score}, value={value}")

        if len(values) == 0:
            return None  # No matching keys so no option at all
        for score, _ in values.values():
            if score < self.threshold:
                return None
        return values

    def best_sample(self, other: "Condition", key: str, eps: float = 1e-15):
        k_max = self.max_state(other, key)
        p1 = self.secure_mix_parameters(key, k_max)
        p2 = other.secure_mix_parameters(key, k_max)

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

    def secure_mix_parameters(self, key: str, k_max: int, eps: float = 1e-15) -> dict:
        p = self.models[key].get_parameters().copy()
        pis = p["measurement"]["state"]["pis"]
        padded = np.full((pis.shape[0], k_max + 1), eps, dtype=np.float32)
        padded[:, : pis.shape[1]] = pis
        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        p["measurement"]["state"]["pis"] = padded
        return p

    def max_state(self, other: "Condition", key: str) -> int:
        s1_max = max(self.model_states[key])
        s2_max = max(other.model_states[key])
        return max(s1_max, s2_max)

    def containment_score(self, other: "Condition", key: str):
        """How much of others mass falls inside selfs distribution."""
        k_max = self.max_state(other, key)
        p1 = self.secure_mix_parameters(key, k_max)
        p2 = other.secure_mix_parameters(key, k_max)

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

    def score_single(
        self, sample: np.ndarray, key: str, eps: float = 1e-15
    ) -> tuple[float, bool]:
        """Score a single sample under a StepMix model. Returns [0,1]."""
        p = self.secure_mix_parameters(key, max(self.model_states[key]))
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
        valid = score >= self.threshold
        return score, valid

    def kl_variational_paper(self, other: "Condition", key: str, eps: float = 1e-15):
        k_max = self.max_state(other, key)
        p1 = self.secure_mix_parameters(key, k_max)
        p2 = other.secure_mix_parameters(key, k_max)

        kl = 0.0
        for i in range(len(p1["weights"])):
            w_i = p1["weights"][i]
            mu1 = p1["measurement"]["pose"]["means"][i]
            var1 = p1["measurement"]["pose"]["covariances"][i]
            cat1 = p1["measurement"]["state"]["pis"][i]

            # Numerator: self-overlap of component i with its OWN model
            log_sum_m1 = 0.0
            for k in range(len(p1["weights"])):
                w_k = p1["weights"][k]
                mu_k = p1["measurement"]["pose"]["means"][k]
                var_k = p1["measurement"]["pose"]["covariances"][k]
                cat_k = p1["measurement"]["state"]["pis"][k]

                kl_self = 0.5 * np.sum(
                    np.log(var1)
                    - np.log(var_k)
                    + var_k / var1
                    + (mu_k - mu1) ** 2 / var1
                    - 1
                )
                cat1_safe = np.clip(cat1, eps, 1)
                cat_k_safe = np.clip(cat_k, eps, 1)
                kl_cat_self = np.sum(
                    cat1_safe * (np.log(cat1_safe) - np.log(cat_k_safe))
                )
                log_sum_m1 += w_k * np.exp(-(kl_self + kl_cat_self))

            # Denominator: cross-overlap with model 2
            log_sum_m2 = 0.0
            for j in range(len(p2["weights"])):
                w_j = p2["weights"][j]
                mu2 = p2["measurement"]["pose"]["means"][j]
                var2 = p2["measurement"]["pose"]["covariances"][j]
                cat2 = p2["measurement"]["state"]["pis"][j]

                kl_gauss = 0.5 * np.sum(
                    np.log(var2)
                    - np.log(var1)
                    + var1 / var2
                    + (mu1 - mu2) ** 2 / var2
                    - 1
                )
                cat2_safe = np.clip(cat2, eps, 1)
                kl_cat = np.sum(cat1_safe * (np.log(cat1_safe) - np.log(cat2_safe)))
                log_sum_m2 += w_j * np.exp(-(kl_gauss + kl_cat))

            kl += w_i * np.log(log_sum_m1 / log_sum_m2)
        return np.exp(-kl)
