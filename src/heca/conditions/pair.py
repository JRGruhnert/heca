from pathlib import Path

from matplotlib import pyplot as plt
import numpy as np

from heca.conditions.condition import Condition

# Condition saves raw datapoints per entity


class ConPair:
    def __init__(
        self,
        label: str,
        pre: Condition,
        post: Condition,
        threshold: float,
    ):
        self.label = label
        self.pre = pre
        self.post = post
        self.threshold = threshold

    @property
    def elabels(self) -> set[str]:
        # Convienience. elbales should never be different
        assert self.pre.elabels == self.post.elabels
        return self.pre.elabels

    @classmethod
    def merge(
        cls,
        label: str,
        a: "ConPair",
        b: "ConPair",
        n_samples: int,
        threshold: float,
    ) -> "ConPair":
        pre_max, post_max = cls.make_max_components(a, b)
        pre_data = cls._merge_data(a.pre, b.pre)
        post_data = cls._merge_data(a.post, b.post)
        pre = Condition(
            "pre",
            pre_data,
            pre_max,
            n_samples,
            threshold,
        )
        post = Condition(
            "post",
            post_data,
            post_max,
            n_samples,
            threshold,
        )
        return cls(label, pre, post, threshold)

    @classmethod
    def make_max_components(cls, a: "ConPair", b: "ConPair") -> tuple[int, int]:
        pre_max = a.pre._max_components + b.pre._max_components
        post_max = a.post._max_components + b.post._max_components
        return pre_max, post_max

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
        print(self.label, other.label)
        mat = np.zeros((2, 2))
        for i, c1 in enumerate([self.pre, self.post]):
            for j, c2 in enumerate([other.pre, other.post]):
                if key not in c1.model_states or key not in c2.model_states:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = c2.containment_score(c1, key)
        return mat

    def compute_sim(self, other: "ConPair") -> dict[str, np.ndarray]:
        sim_rating = {}
        for el in self.elabels.intersection(other.elabels):
            forward = self.calculate_sim_matrix(other, el)
            backward = other.calculate_sim_matrix(self, el)
            sim_rating[el] = np.stack((forward, backward), axis=0)
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

    def can_merge(self, other: "ConPair", path: Path | None = None) -> bool:
        sim_rating = self.compute_sim(other)
        if path is not None:
            self.plot_similarity(sim_rating, other, path)
        merge = self.evaluate_merge(sim_rating)
        print(f"{self.label} and {other.label} merge: {merge}")
        return merge

    def mcheck(self, mat: np.ndarray):
        return np.all(mat >= self.threshold)

    def evaluate_merge(self, sim_rating: dict[str, np.ndarray]) -> bool:
        mat = np.stack(list(sim_rating.values()), axis=0)
        mat = np.nan_to_num(mat, nan=1.0)  # nan values should be ignored
        if self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 1, 0]):
            return True  # pre0 ↔ post1 (bidirectional equivalence)
        elif self.mcheck(mat[:, 0, 1, 0]) and self.mcheck(mat[:, 1, 1, 0]):
            return True  # post0 ⊆ pre1 AND post1 ⊆ pre0 (sequential)
        elif self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 0, 1]):
            return True  # pre0 in post1 and pre1 in post0
        elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 1, 1]):
            return True  # pre0 in pre1 and post0 in post1
        elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 1, 1]):
            return True  # pre1 in pre0 and post1 in post0
        elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 0, 1]):
            return True  # pre0 in pre1 and pre0 in post1
        elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 0, 1]):
            return True  # pre1 in pre0 and pre1 in post0
        return False
