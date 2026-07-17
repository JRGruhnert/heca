from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix
from scipy.optimize import minimize
from heca.misc.entity import STATE_LOGIT_BASELINE
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

    @property
    def parameters(
        self,
    ) -> dict[str, list[np.ndarray]]:
        """
        Extracts node features for the StepMix tree-graph.

        Returns:
            dict: Keys are model names. Values are lists of flat feature arrays,
                one per component. Each feature array has shape (13+K,) where:
                - [0:3] = μ_pos (position means)
                - [3:6] = log(σ_pos) (log standard deviations of position)
                - [6:10] = quaternion [w, x, y, z] (rotation mean)
                - [10:13] = log(σ_rot) (log standard deviations of rotation in tangent space)
                - [13:] = logits (state logits, unnormalized)

        This format matches the output of :meth:`Entity.gnn_format` and is consumed
        by :meth:`EdgeSet.compute_edge_feats`.
        """

        # NOTE: ASSUMES MODELS USE DIAG MODE
        result: dict[str, list[np.ndarray]] = {}

        for model_name, mix in self.models.items():
            # Extract raw parameters from the fitted StepMix model
            params = mix.get_parameters()
            weights = params["weights"]  # shape: (K,)
            means = params["measurement"]["pose"]["means"]  # shape: (K, 7)
            covariances = params["measurement"]["pose"]["covariances"]  # shape: (K, 7)
            pis = params["measurement"]["state"]["pis"]  # shape: (K, total_outcomes)

            n_components = weights.shape[0]  # type: ignore
            n_observed_outcomes = pis.shape[1]
            total_states = self.model_states[model_name]

            if total_states > n_observed_outcomes:
                observed_states = np.sort(
                    np.unique(self._data_raw[model_name][:, 7].astype(int))
                )
            else:
                observed_states = None

            components_list: list[np.ndarray] = []
            for i in range(n_components):
                # --- 1. Position mean (3) ---
                mu_pos = means[i, :3]  # [3]

                # --- 2. Position log-std (3) ---
                # Convert variance to log standard deviation for unconstrained optimization.
                lstd_pos = 0.5 * np.log(covariances[i, :3] + 1e-8)  # [3]

                # --- 3. Rotation quaternion mean (4) ---
                quat = means[i, 3:7]  # [4] - [w, x, y, z]
                quat /= np.linalg.norm(quat)  # norm
                # Quaternion.normalize
                # --- 4. Rotation log-std (3) ---
                # Rotation uncertainty has 3 DOF in tangent space (axis-angle).
                # We take the first 3 of the 4 quaternion-component variances.
                lstd_rot = 0.5 * np.log(covariances[i, 3:6] + 1e-8)  # [3]

                # --- 5. State logits (K) ---
                # Convert probabilities → logits (unconstrained).
                logits_compact = np.log(pis[i, :] + 1e-8)  # [C]
                if observed_states is not None:
                    logits = np.full(
                        sorted(total_states), STATE_LOGIT_BASELINE, dtype=np.float32
                    )
                    for j, state_idx in enumerate(observed_states):
                        logits[state_idx] = logits_compact[j]
                else:
                    logits = logits_compact

                # Concatenate into flat feature vector [13 + K]
                feature = np.concatenate([mu_pos, lstd_pos, quat, lstd_rot, logits])

                components_list.append(feature)

            result[model_name] = components_list

        return result

    @property
    def model_states(self) -> dict[str, set[int]]:
        states: dict[str, set[int]] = {}

        for key, values in self._data_raw.items():
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

    def plot(self, path: Path):
        self._plot_bic(path)
        self._plot_mix(path)

    def _plot_bic(self, path: Path):
        ks = range(1, self._max_components + 1)

        plt.figure(figsize=(8, 5))

        for name, bic in self._bics.items():
            plt.plot(ks, bic, marker="o", label=f"{name}")

        plt.xlabel("Number of latent classes")
        plt.ylabel("BIC")
        plt.title("Model selection")
        plt.grid(True)
        plt.legend()
        plt.savefig(path / f"{self.label}_bic.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _plot_mix(self, path: Path):
        pass

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

    def best_sample(self, con: "Condition", key: str):
        m1, s1 = Condition.mix_with_states(self, key)
        m2, s2 = Condition.mix_with_states(con, key)
        p1 = m1.get_parameters()
        p2 = m2.get_parameters()
        weights1 = p1["weights"]
        weights2 = p2["weights"]
        means1 = p1["measurement"]["pose"]["means"]
        means2 = p2["measurement"]["pose"]["means"]
        vars1 = p1["measurement"]["pose"]["covariances"]
        vars2 = p2["measurement"]["pose"]["covariances"]
        state1 = p1["measurement"]["state"]["pis"]
        state2 = p2["measurement"]["state"]["pis"]

        # Align categorical distributions to the union of observed states
        target = sorted(s1 | s2)
        padded1 = self._pad_cat_probs(state1, s1, target)
        padded2 = self._pad_cat_probs(state2, s2, target)
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
                cat_prod = padded1[i] * padded2[j]  # element-wise over aligned states
                best_s_idx = int(np.argmax(cat_prod))
                best_state = target[best_s_idx]
                log_cat = np.log(np.clip(cat_prod[best_s_idx], 1e-12, None))
                score = np.log(weights1[i]) + np.log(weights2[j]) + log_norm + log_cat
                results.append(
                    {
                        "score": score,
                        "pose": mean,
                        "state": best_state,
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

    @staticmethod
    def _pad_cat_probs(
        pis_compact: np.ndarray,
        observed_states: set[int],
        target_states: list[int],
        eps: float = 1e-12,
    ) -> np.ndarray:
        """Expand compact categorical probs to a full target state space.

        Args:
            pis_compact: Shape (n_components, n_observed_outcomes)
            observed_states: Sorted set of observed state indices (e.g. {0, 2})
            target_states: Full list of states to pad to (e.g. [0, 1, 2])
            eps: Small value for missing entries

        Returns:
            Padded probs of shape (n_components, len(target_states))
        """
        n_components = pis_compact.shape[0]
        n_target = len(target_states)
        padded = np.full((n_components, n_target), eps, dtype=np.float32)

        observed_list = sorted(observed_states)
        for j, state_idx in enumerate(observed_list):
            idx = target_states.index(state_idx)
            padded[:, idx] = pis_compact[:, j]

        # Renormalize so probabilities sum to 1
        padded /= padded.sum(axis=1, keepdims=True)
        return padded

    @staticmethod
    def mix_with_states(con: "Condition", key: str) -> tuple[StepMix, set[int]]:
        return con.models[key], con.model_states[key]

    def kl_variational_paper(self, con: "Condition", key: str):
        m1, s1 = Condition.mix_with_states(self, key)
        m2, s2 = Condition.mix_with_states(con, key)
        p1 = m1.get_parameters()
        p2 = m2.get_parameters()

        # Align categorical distributions to a common state space
        if s1 is not None and s2 is not None:
            target = sorted(s1 | s2)  # union
            padded_cat1 = self._pad_cat_probs(
                p1["measurement"]["state"]["pis"], s1, target
            )
            padded_cat2 = self._pad_cat_probs(
                p2["measurement"]["state"]["pis"], s2, target
            )
        else:
            padded_cat1 = p1["measurement"]["state"]["pis"]
            padded_cat2 = p2["measurement"]["state"]["pis"]

        kl = 0.0
        for i in range(len(p1["weights"])):
            w1 = p1["weights"][i]
            mu1 = p1["measurement"]["pose"]["means"][i]
            var1 = p1["measurement"]["pose"]["covariances"][i]
            cat1 = padded_cat1[i]

            log_sum_m1 = 0.0
            log_sum_m2 = 0.0
            for j in range(len(p2["weights"])):
                w2 = p2["weights"][j]
                mu2 = p2["measurement"]["pose"]["means"][j]
                var2 = p2["measurement"]["pose"]["covariances"][j]
                cat2 = padded_cat2[j]

                kl_gauss = 0.5 * np.sum(
                    np.log(var2)
                    - np.log(var1)
                    + var1 / var2
                    + (mu1 - mu2) ** 2 / var2
                    - 1
                )
                cat1_safe = np.clip(cat1, 1e-12, 1)
                cat2_safe = np.clip(cat2, 1e-12, 1)
                kl_cat = np.sum(cat1_safe * (np.log(cat1_safe) - np.log(cat2_safe)))
                total_kl = kl_gauss + kl_cat

                log_sum_m1 += w1 * np.exp(-total_kl)  # numerator: w_i * exp(-KL)
                log_sum_m2 += w2 * np.exp(-total_kl)  # denominator: v_j * exp(-KL)

            kl += w1 * np.log(log_sum_m1 / log_sum_m2)

        return np.exp(-kl)

    def score_single(self, sample: np.ndarray, key: str) -> tuple[float, bool]:
        """Score a single sample under a StepMix model. Returns [0,1]."""
        m, s = Condition.mix_with_states(self, key)
        sample_pose = sample[:7]
        sample_state = sample[-1]
        params = m.get_parameters()
        best_log_prob = -np.inf
        for k in range(len(params["weights"])):
            mu = params["measurement"]["pose"]["means"][k]
            var = params["measurement"]["pose"]["covariances"][k]
            pis = params["measurement"]["state"]["pis"][k]

            # Gaussian
            log_gauss = -0.5 * np.sum(
                np.log(2 * np.pi * var) + (sample_pose - mu) ** 2 / var
            )

            # Categorical
            if s is not None:
                sorted_states = sorted(s)
                try:
                    idx = sorted_states.index(sample_state)
                    state_prob = pis[idx]
                except ValueError:
                    state_prob = 1e-12  # unseen state
            else:
                # Fallback: assumes pis is indexed by state value directly
                state_prob = pis[sample_state] if sample_state < len(pis) else 1e-12

            log_cat = np.log(np.clip(state_prob, 1e-12, 1))
            score = np.log(params["weights"][k]) + log_gauss + log_cat

            if score > best_log_prob:
                best_log_prob = score

        score = np.exp(best_log_prob)
        valid = score < self.threshold
        return score, valid
