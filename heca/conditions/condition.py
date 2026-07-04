from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix

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
        return {k: float(self.model[k].score(v)) for k, v in self.samples.items()}

    @property
    def raw_self_scores(self) -> dict[str, float]:
        return {k: float(self.model[k].score(v)) for k, v in self.data_raw.items()}

    @property
    def model(self) -> dict[str, StepMix]:
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
