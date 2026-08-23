from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from heca.conditions.condition import Condition
from heca.data.entity import Entity


class ConPair:
    # Floor for the pooled std so constant dimensions (e.g. a state column that
    # never varies in one condition) do not produce division by zero.
    _CHANGE_EPS = 1e-6

    def __init__(self, label: str, pre: Condition, post: Condition):
        self.label = label
        self.pre = pre
        self.post = post
        self._change_scores = self._compute_change_scores()

    def _compute_change_scores(self) -> dict[str, float]:
        scores: dict[str, float] = {}
        for key in set(self.pre.data_raw).intersection(set(self.post.data_raw)):
            pre = np.asarray(self.pre.data_raw[key], dtype=np.float64)
            post = np.asarray(self.post.data_raw[key], dtype=np.float64)
            pooled_std = np.sqrt((pre.var(axis=0) + post.var(axis=0)) / 2.0)
            pooled_std = np.maximum(pooled_std, self._CHANGE_EPS)
            z = np.abs(post.mean(axis=0) - pre.mean(axis=0)) / pooled_std
            scores[key] = float(z.mean())
        return scores

    @property
    def change_scores(self) -> dict[str, float]:
        if not hasattr(self, "_change_scores"):
            self._change_scores = self._compute_change_scores()
        return self._change_scores

    # @classmethod
    # def merge(
    #     cls,
    #     label: str,
    #     a: "ConPair",
    #     b: "ConPair",
    #     n_samples: int,
    #     threshold: float,
    # ) -> "ConPair":
    #     pre_max, post_max = cls.make_max_components(a, b)
    #     pre_data = cls._merge_data(a.pre, b.pre)
    #     post_data = cls._merge_data(a.post, b.post)
    #     pre = Condition(
    #         "pre",
    #         pre_data,
    #         pre_max,
    #         n_samples,
    #     )
    #     post = Condition(
    #         "post",
    #         post_data,
    #         post_max,
    #         n_samples,
    #         threshold,
    #     )
    #     return cls(label, pre, post, threshold)

    @classmethod
    def make(
        cls,
        tag,
        pre_data: dict[str, np.ndarray],
        post_data: dict[str, np.ndarray],
        entities: dict[str, Entity],
    ) -> "ConPair":
        pre = Condition("pre", pre_data, entities)
        post = Condition("post", post_data, entities)
        return cls(f"{tag}", pre, post)

    @classmethod
    def _merge_data(
        cls,
        c1: Condition,
        c2: Condition,
    ) -> dict[str, np.ndarray]:
        result = c1.data_raw.copy()
        for k, v in c2.data_raw.items():
            result[k] = np.concatenate((result[k], v), axis=0) if k in result else v
        return result

    def plot(self, path: Path):
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        self.pre.plot(plot_path, self.label)
        self.post.plot(plot_path, self.label)

    def calculate_sim_matrix(self, other: "ConPair", key: str) -> np.ndarray:
        mat = np.zeros((2, 2))
        for i, c1 in enumerate([self.pre, self.post]):
            for j, c2 in enumerate([other.pre, other.post]):
                if key not in c1.entities or key not in c2.entities:
                    mat[i, j] = np.nan
                else:
                    up1 = c2.models[key].get_parameters().copy()
                    up2 = c1.models[key].get_parameters().copy()
                    mat[i, j] = c1.entities[key].containment_score(up1, up2)
        return mat

    def compute_sim(self, other: "ConPair") -> dict[str, np.ndarray]:
        sim_rating = {}
        for key in set(self.pre.entities).intersection(set(other.pre.entities)):
            forward = self.calculate_sim_matrix(other, key)
            backward = other.calculate_sim_matrix(self, key)
            sim_rating[key] = np.stack((forward, backward), axis=0)
        return sim_rating

    def plot_similarity(
        self,
        sim_rating: dict[str, np.ndarray],
        other: "ConPair",
        path: Path,
    ):
        entities = list(sim_rating.keys())
        n = len(entities)

        fig, axes = plt.subplots(2, n, figsize=(3.5 * n, 7), squeeze=False)

        xticklabels = ["pre", "post"]
        yticklabels = ["pre", "post"]
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("red")
        for c, entity in enumerate(entities):
            for r in [0, 1]:
                ax = axes[r, c]
                mat = sim_rating[entity][r]

                im = ax.imshow(
                    mat,
                    cmap=cmap,
                    vmin=0.0,
                    vmax=1.0,
                )

                ax.set_xticks([0, 1])
                ax.set_xticklabels(xticklabels)

                ax.set_yticks([0, 1])
                ax.set_yticklabels(yticklabels)

                if r == 0:
                    ax.set_title(entity)

                if c == 0:
                    ax.set_ylabel(self.label if r == 0 else other.label)
                    ax.set_xlabel(other.label if r == 0 else self.label)

                # annotate values
                for i in range(2):
                    for j in range(2):
                        value = mat[i, j]
                        text = "" if np.isnan(value) else f"{value:.2f}"
                        ax.text(
                            j,
                            i,
                            text,
                            ha="center",
                            va="center",
                            color="red",
                            fontsize=9,
                        )

        fig.colorbar(im, ax=axes, label="Similarity", shrink=0.8)
        fig.suptitle(f"{other.label} ↔ {self.label} similarity", fontsize=16)
        plt.savefig(path / "plots" / f"sim_{other.label}_{self.label}.png", dpi=300)
        plt.close(fig)

    # def can_merge(self, other: "ConPair", path: Path | None = None) -> bool:
    #     sim_rating = self.compute_sim(other)
    #     if path is not None:
    #         self.plot_similarity(sim_rating, other, path)
    #     return self.evaluate_sim(sim_rating)

    # def mcheck(self, mat: np.ndarray):
    #     return np.all(mat >= self.threshold)

    # def evaluate_sim(self, sim_rating: dict[str, np.ndarray]) -> bool:
    #     mat = np.stack(list(sim_rating.values()), axis=0)
    #     mat = np.nan_to_num(mat, nan=1.0)  # nan values should be ignored
    #     if self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 1, 0]):
    #         return True  # pre0 ↔ post1 (bidirectional equivalence)
    #     elif self.mcheck(mat[:, 0, 1, 0]) and self.mcheck(mat[:, 1, 1, 0]):
    #         return True  # post0 ⊆ pre1 AND post1 ⊆ pre0 (sequential)
    #     elif self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 0, 1]):
    #         return True  # pre0 in post1 and pre1 in post0
    #     elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 1, 1]):
    #         return True  # pre0 in pre1 and post0 in post1
    #     elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 1, 1]):
    #         return True  # pre1 in pre0 and post1 in post0
    #     elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 0, 1]):
    #         return True  # pre0 in pre1 and pre0 in post1
    #     elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 0, 1]):
    #         return True  # pre1 in pre0 and pre1 in post0
    #     return False

    @property
    def entities(self) -> dict[str, Entity]:
        return self.pre.entities
