from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix
from scipy.optimize import minimize
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
    ) -> dict[str, list[np.ndarray]]:
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

        result: dict[str, list[np.ndarray]] = {}

        for key in self.models.keys():
            # Extract raw parameters from the fitted StepMix model
            params = self.secure_mix_parameters(key, len(entities[key].cfg.states))
            weights = params["weights"]  # shape: (N,)
            means = params["measurement"]["pose"]["means"]  # shape: (N, 7)
            covariances = params["measurement"]["pose"]["covariances"]  # shape: (N, 7)
            pis = params["measurement"]["state"]["pis"]  # shape: (K, total_outcomes)
            N = len(weights)

            feat_list: list[np.ndarray] = []
            for i in range(N):
                mu_pos = means[i, :3]  # [3]
                quat = means[i, 3:7]  # [4] - [w, x, y, z]
                quat /= np.linalg.norm(quat)  # norm
                lstd_pos = 0.5 * np.log(covariances[i, :3] + eps)  # [3]
                lstd_rot = 0.5 * np.log(covariances[i, 3:6] + eps)  # [3]
                logits = np.log(pis[i, :])  # [K]
                feature = np.concatenate([mu_pos, lstd_pos, quat, lstd_rot, logits])
                feat_list.append(feature)

            result[key] = feat_list

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
            path / f"{label}_{self.label}_bic.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def make_subgoal(
        self, other: "Condition"
    ) -> dict[str, tuple[float, np.ndarray]] | None:
        values = {}
        for key in self.elabels.intersection(other.elabels):
            score = self.kl_variational_paper(other, key)
            value = self.best_sample(other, key)
            values[key] = (score, value)

        if len(values) == 0:
            return None  # No matching keys so no option at all
        for score, _ in values.values():
            if score < self.threshold:
                return None
        return values

    def best_sample(self, other: "Condition", key: str):
        p1 = self.secure_mix_parameters(key)
        p2 = other.secure_mix_parameters(key)

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
                best_s_idx = int(np.argmax(cat_prod))
                log_cat = np.log(np.clip(cat_prod[best_s_idx], 1e-12, None))
                score = np.log(weights1[i]) + np.log(weights2[j]) + log_norm + log_cat
                results.append(
                    {
                        "score": score,
                        "pose": mean,
                        "state": best_s_idx,
                        "component1": i,
                        "component2": j,
                    }
                )

        results.sort(key=lambda r: r["score"], reverse=True)
        pose = results[0]["pose"]
        state = results[0]["state"]
        assert isinstance(pose, np.ndarray)
        return np.concatenate([pose, [state]])

    def refine_pose(self, initial_pose, m1: StepMix, m2: StepMix):

        def neg_log_product(x):
            # Minimize the negative sum of log-densities
            return -(m1.log_prob(x) + m2.log_prob(x))

        # Gradient is automatically computed via finite differences or autograd
        result = minimize(neg_log_product, initial_pose, method="L-BFGS-B")
        return result.x

    def secure_mix_parameters(
        self, key: str, K: int | None = None, eps: float = 1e-12
    ) -> dict:
        self.models[key]
        self.model_states[key]
        p = self.models[key].get_parameters()

        padded_cat1 = self._pad_cat_probs(p["measurement"]["state"]["pis"], s1, target)
        n_components = pis_compact.shape[0]
        n_target = len(target_states)
        padded = np.full((n_components, n_target), eps, dtype=np.float32)

        for j, state_val in enumerate(state_values):
            idx = target_states.index(state_val)
            padded[:, idx] = pis_compact[:, j]
        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        return padded
        return p

    def containment_score(self, other: "Condition", key: str):
        """How much of B's mass falls inside A's distribution."""
        p1 = self.secure_mix_parameters(key)
        p2 = other.secure_mix_parameters(key)

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
                if key == "button_1":
                    print(f"i={i}, j={j}")
                    print(f"Target cat: {cat1}")
                    print(f"Source cat: {cat2}")
                    print(f"Overlap_cat: {overlap_cat}")
                    print(f"Peak_target: {peak_target}")
                    print(f"Cat_score: {cat_score}")
                score += w_i * w_j * gauss_rel * cat_score

        return score  # [0, 1]

    def score_single(self, sample: np.ndarray, key: str) -> tuple[float, bool]:
        """Score a single sample under a StepMix model. Returns [0,1]."""
        p = self.secure_mix_parameters(key)
        sample_pose = sample[:7]
        sample_state = sample[-1]
        best_log_prob = -np.inf
        for k in range(len(p["weights"])):
            mu = p["measurement"]["pose"]["means"][k]
            var = p["measurement"]["pose"]["covariances"][k]
            pis = p["measurement"]["state"]["pis"][k]

            # Gaussian
            log_gauss = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (sample_pose - mu) ** 2 / var
            )

            # Categorical
            state_prob = pis[sample_state] if sample_state < len(pis) else 1e-12

            log_cat = np.log(np.clip(state_prob, 1e-12, 1))
            score = np.log(p["weights"][k]) + log_gauss + log_cat

            if score > best_log_prob:
                best_log_prob = score

        score = np.exp(best_log_prob)
        valid = score < self.threshold
        return score, valid

    def kl_variational_paper(self, other: "Condition", key: str):
        p1 = self.secure_mix_parameters(key)
        p2 = other.secure_mix_parameters(key)

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
                cat1_safe = np.clip(cat1, 1e-12, 1)
                cat_k_safe = np.clip(cat_k, 1e-12, 1)
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
                cat2_safe = np.clip(cat2, 1e-12, 1)
                kl_cat = np.sum(cat1_safe * (np.log(cat1_safe) - np.log(cat2_safe)))
                log_sum_m2 += w_j * np.exp(-(kl_gauss + kl_cat))

            kl += w_i * np.log(log_sum_m1 / log_sum_m2)
            if key == "window_handle":
                print(f"Component {i}:")
                print(f"mu1={mu1}")
                print(f"mu2={mu2}")
                print(f"var1={var1}")
                print(f"var2={var2}")
                print(f"cat1={cat1}, cat2={cat2}")
                print(f"log_sum_m1={log_sum_m1:.4f}")
                print(f"log_sum_m2={log_sum_m2:.4f}")
        if key == "window_handle":
            print(f"Total KL={kl:.4f}, score={np.exp(-kl):.4f}")
        return np.exp(-kl)
