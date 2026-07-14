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
        pre_data, pre_states = cls._merge_data(a.pre, b.pre)
        post_data, post_states = cls._merge_data(a.post, b.post)
        pre = Condition(
            "pre",
            pre_data,
            pre_max,
            n_samples,
            pre_states,
            threshold,
        )
        post = Condition(
            "post",
            post_data,
            post_max,
            n_samples,
            post_states,
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
    ) -> tuple[dict[str, np.ndarray], dict[str, int]]:
        result = c1.data_raw.copy()
        for k, v in c2.data_raw.items():
            result[k] = np.concatenate((result[k], v), axis=0) if k in result else v
        return result, {**c1._state_counts, **c2._state_counts}

    def plot(self, path: Path):
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        self.pre.plot(plot_path)
        self.post.plot(plot_path)

    def calculate_sim_matrix(
        self, cp1: "ConPair", cp2: "ConPair", key: str
    ) -> np.ndarray:
        mat = np.zeros((2, 2))
        for i, cp1con in enumerate([cp1.pre, cp1.post]):
            for j, cp2con in enumerate([cp2.pre, cp2.post]):
                if key not in cp1con.model_states or key not in cp2con.model_states:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = cp2con.kl_variational_paper(cp1con, key)
        return mat

    def compute_sim(self, other: "ConPair") -> dict[str, np.ndarray]:
        sim_rating = {}
        for el in self.elabels.intersection(other.elabels):
            forward = self.calculate_sim_matrix(self, other, el)
            backward = self.calculate_sim_matrix(other, self, el)
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
        return self.evaluate_merge(sim_rating)

    def mcheck(self, mat: np.ndarray):
        return np.all(mat >= self.threshold)

    def evaluate_merge(self, sim_rating: dict[str, np.ndarray]) -> bool:
        mat = np.stack(list(sim_rating.values()), axis=0)
        mat = np.nan_to_num(mat, nan=1.0)  # nan values should be ignored
        if self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 1, 0]):
            return True  # pre0 = post1
        elif self.mcheck(mat[:, 1, 0, 1]) and self.mcheck(mat[:, 0, 1, 0]):
            return True  # pre1 = post0
        elif self.mcheck(mat[:, 0, 0, 1]) and self.mcheck(mat[:, 1, 0, 1]):
            return True  # pre0 <= post1 and pre1 <= post0
        elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 1, 1]):
            return True  # pre0 <= pre1 and post0 <= post1
        elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 1, 1]):
            return True  # pre1 <= pre0 and post1 <= post0
        elif self.mcheck(mat[:, 0, 0, 0]) and self.mcheck(mat[:, 0, 0, 1]):
            return True  # pre0 <= pre1 and pre0 <= post1
        elif self.mcheck(mat[:, 1, 0, 0]) and self.mcheck(mat[:, 1, 0, 1]):
            return True  # pre1 <= pre0 and pre1 <= post0
        return False
