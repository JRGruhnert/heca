from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from stepmix import StepMix
from heca.misc import logger
from heca.data.entity import Entity


class Condition:
    def __init__(
        self, label: str, data: dict[str, np.ndarray], entities: dict[str, Entity]
    ):
        self._data_raw = data
        self._entities = entities
        self.label = label

        self._model, self._bics = self._fit_model()

    def comp_features(self) -> dict[str, np.ndarray]:
        result: dict[str, np.ndarray] = {}
        for key in self.models.keys():
            up = self.models[key].get_parameters().copy()
            result[key] = self.entities[key].comp_feature(up)
        return result

    @property
    def data_raw(self) -> dict[str, np.ndarray]:
        return self._data_raw

    # @property
    # def samples(self) -> dict[str, np.ndarray]:
    #     return self._samples

    # @property
    # def sample_self_scores(self) -> dict[str, float]:
    #     return {k: float(self.models[k].score(v)) for k, v in self.samples.items()}

    @property
    def raw_self_scores(self) -> dict[str, float]:
        return {
            k: float(self.models[k].score(self.entities[k].model_value(v)))
            for k, v in self.data_raw.items()
        }

    @property
    def models(self) -> dict[str, StepMix]:
        return self._model

    @property
    def entities(self) -> dict[str, Entity]:
        return self._entities

    def _fit_model(self) -> tuple[
        dict[str, StepMix],
        # dict[str, np.ndarray],
        dict[str, list[float]],
    ]:

        models: dict[str, StepMix] = {}
        # samples: dict[str, np.ndarray] = {}
        bics: dict[str, list[float]] = {}

        for key, values in self.data_raw.items():
            best_model = None
            best_bic = np.inf
            bic_values: list[float] = []

            for k in range(1, self.entities[key].cfg.max_fit_components + 1):
                model = StepMix(
                    n_components=k,
                    measurement=self.entities[key].measurement,  # type: ignore
                    verbose=False,
                    progress_bar=0,
                )

                model.fit(self.entities[key].model_value(values))
                n_outcomes = model.get_parameters()["measurement"]["state"][
                    "pis"
                ].shape[1]
                if n_outcomes > self.entities[key].cfg.n_states:
                    raise ValueError(
                        f"Entity '{key}': fitted categorical has {n_outcomes} "
                        f"outcomes but cfg.n_states="
                        f"{self.entities[key].cfg.n_states}. Increase n_states "
                        "in the entity config to match the distinct states in "
                        "the data, otherwise the GNN features silently drop "
                        "states."
                    )
                bic = model.bic(self.entities[key].model_value(values))
                bic_values.append(bic)

                if bic < best_bic:
                    best_bic = bic
                    best_model = model

            assert best_model is not None
            models[key] = best_model
            # samples[key] = best_model.sample(self._n_samples)[0]
            bics[key] = bic_values

        return models, bics

    # def score(self, x: dict[str, np.ndarray]) -> dict[str, float]:
    #     scores: dict[str, float] = {}
    #     for key, model in self.models.items():
    #         raw = model.score(x)
    #         delta = raw - self.sample_self_scores[key]
    #         clipped = np.minimum(delta, 0)  # we just care for negative deltas
    #         scores[key] = np.exp(clipped)
    #     return scores

    def plot(self, path: Path, label: str):
        ks = range(1, 10 + 1)
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
        self, other: "Condition"
    ) -> dict[str, tuple[float, np.ndarray]] | None:
        values = {}
        for key in set(self.entities).intersection(set(other.entities)):
            up1 = self.models[key].get_parameters().copy()
            up2 = other.models[key].get_parameters().copy()
            score = self.entities[key].containment_score(up1, up2)
            # Threshold-free: connect when at least some of the goal model's
            # mass lies inside the pre model's z_quantile ellipsoid (the score
            # is calibrated by cfg.z_quantile, not a tuned cfg.threshold).
            if score <= 0.0:
                return None
            value = self.entities[key].best_sample(up1, up2)
            values[key] = (score, value)
            logger.debug(f"{key}: score={score}, value={value}")

        if len(values) == 0:
            return None  # No matching keys so no option at all
        return values

    def scores(self, other: "Condition") -> dict[str, float]:
        """Containment score + threshold for every shared entity.

        Unlike ``make_subgoal``, this does not early-exit, so it can be used to
        inspect why a connection was (not) formed.
        """
        result = {}
        for key in set(self.entities).intersection(set(other.entities)):
            up1 = self.models[key].get_parameters().copy()
            up2 = other.models[key].get_parameters().copy()
            score = self.entities[key].containment_score(up1, up2)
            result[key] = score
        return result
