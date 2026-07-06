from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix

from heca.misc.quaternion import Quaternion

# Condition saves raw datapoints per entity


class Condition:
    def __init__(
        self,
        label: str,
        data: dict[str, np.ndarray],
        max_components: int,
        n_samples: int,
    ):
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
    def model_parameters(
        self,
    ) -> dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
        """
        Extracts node features for the StepMix tree-graph.

        Returns:
            dict: Keys are model names. Values are lists of tuples, one per component.
                Each tuple contains:
                - pos_feature (np.ndarray): [μ_x, μ_y, μ_z, σ_x, σ_y, σ_z] (shape: 6)
                - rot_feature (np.ndarray): [6D_rot_mean, σ_w, σ_x, σ_y, σ_z] (shape: 10)
                - state_feature (np.ndarray): Probabilities for each category (shape: total_outcomes)
        """
        result: dict[str, list[tuple[np.ndarray, np.ndarray, np.ndarray]]] = {}

        for model_name, mix in self.models.items():
            # 1. Extract raw parameters from the fitted StepMix model
            params = mix.get_parameters()
            weights = params["weights"]  # shape: (K,)
            means = params["measurement"]["pose"]["means"]  # shape: (K, 7)
            covariances = params["measurement"]["pose"][
                "covariances"
            ]  # shape: (K, 7) - assumes diagonal!
            pis = params["measurement"]["state"]["pis"]  # shape: (K, total_outcomes)

            n_components = weights.shape[0]  # type: ignore
            components_list = []

            for i in range(n_components):
                # --- A. Position Feature (First 3 continuous variables) ---
                pos_mean = means[i, :3]
                pos_std = np.sqrt(covariances[i, :3])  # Standard deviation
                pos_feature = np.concatenate([pos_mean, pos_std])  # Shape: (6,)

                # --- B. Rotation Feature (Last 4 continuous variables: w, x, y, z) ---
                quat_mean = means[i, 3:7]  # [w, x, y, z]
                quat_std = np.sqrt(covariances[i, 3:7])  # Standard deviations

                # Convert the mean quaternion to 6D continuous representation
                rot_6d = Quaternion.quat_to_6d(
                    quat_mean[0], quat_mean[1], quat_mean[2], quat_mean[3]
                )  # Shape: (6,)

                # Concatenate the 6D mean with the 4 standard deviations
                rot_feature = np.concatenate([rot_6d, quat_std])  # Shape: (10,)

                # --- C. State Feature (Categorical probabilities) ---
                # Pis for this specific component. Shape: (total_outcomes,)
                state_feature = pis[i, :]

                # Store the tuple for this component
                components_list.append((pos_feature, rot_feature, state_feature))

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
