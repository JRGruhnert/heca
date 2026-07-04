from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from heca.conditions.condition import Condition
from heca.conditions.pair import ConditionPair


class ConditionAnalyzer:

    def check(self, mat: np.ndarray, ths: float):
        return np.all(mat >= ths)

    def evaluate_merge(self, sim_rating: dict[str, np.ndarray], ths: float) -> bool:
        mat = np.stack(list(sim_rating.values()), axis=0)
        mat = np.nan_to_num(mat, nan=1.0)  # nan values should be ignored

        if self.check(mat[:, 0, 0, 1], ths) and self.check(mat[:, 1, 1, 0], ths):
            return True  # pre0 = post1
        elif self.check(mat[:, 1, 0, 1], ths) and self.check(mat[:, 0, 1, 0], ths):
            return True  # pre1 = post0
        elif self.check(mat[:, 0, 0, 1], ths) and self.check(mat[:, 1, 0, 1], ths):
            return True  # pre0 <= post1 and pre1 <= post0
        elif self.check(mat[:, 0, 0, 0], ths) and self.check(mat[:, 0, 1, 1], ths):
            return True  # pre0 <= pre1 and post0 <= post1
        elif self.check(mat[:, 1, 0, 0], ths) and self.check(mat[:, 1, 1, 1], ths):
            return True  # pre1 <= pre0 and post1 <= post0
        elif self.check(mat[:, 0, 0, 0], ths) and self.check(mat[:, 0, 0, 1], ths):
            return True  # pre0 <= pre1 and pre0 <= post1
        elif self.check(mat[:, 1, 0, 0], ths) and self.check(mat[:, 1, 0, 1], ths):
            return True  # pre1 <= pre0 and pre1 <= post0
        return False

    def evaluate_subgoal(self, sim_rating: dict[str, np.ndarray]) -> np.ndarray:
        raise NotImplementedError

    def calculate_subgoal(
        self, post: Condition, pre: Condition, keys: list[str]
    ) -> dict[str, str]:
        values = {}
        for key in keys:
            inpost = key in post.elabels
            inpre = key in pre.elabels
            if not inpost:
                continue
            elif not inpre:
                pass
                # not in pre -> either set as goal if goal in post distibution or current or sample own
                # 1. goal
                # 2. current
                # 3. sample from pre
            else:
                # 1. Pre sample
                # 2. Post sample
                post_score = self.calculate_score(post, pre, key)
                pre_score = self.calculate_score(pre, post, key)
        return values

    def calculate_sim_matrix(
        self, cp1: ConditionPair, cp2: ConditionPair, key: str
    ) -> np.ndarray:
        mat = np.zeros((2, 2))
        for i, cp1con in enumerate([cp1.pre, cp1.post]):
            for j, cp2con in enumerate([cp2.pre, cp2.post]):
                if not cp1con.model_states[key].issubset(cp2con.model_states[key]):
                    continue
                if key not in cp1con.model_states or key not in cp2con.model_states:
                    mat[i, j] = np.nan
                else:
                    mat[i, j] = self.calculate_score(cp2con, cp1con, key)
        return mat

    def calculate_score(self, a: Condition, b: Condition, key: str) -> float:
        raw = a.model[key].score(b.samples[key])
        delta = raw - a.sample_self_scores[key]
        clipped = np.minimum(delta, 0)  # we just care for negative deltas
        return np.exp(clipped)  # make a score

    def compute_sim(
        self,
        cp1: "ConditionPair",
        cp2: "ConditionPair",
    ) -> dict[str, np.ndarray]:
        sim_rating = {}
        for el in cp1.elabels | cp2.elabels:
            forward = self.calculate_sim_matrix(cp1, cp2, el)
            backward = self.calculate_sim_matrix(cp2, cp1, el)
            sim_rating[el] = np.stack((forward, backward), axis=0)

        return sim_rating

    def plot_similarity(
        self,
        sim_rating: dict[str, np.ndarray],
        cp1: "ConditionPair",
        cp2: "ConditionPair",
        path: Path,
    ):
        entities = list(sim_rating.keys())
        n = len(entities)

        fig, axes = plt.subplots(
            2,
            n,
            figsize=(3.5 * n, 7),
            squeeze=False,
            constrained_layout=True,
        )

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
                    ax.set_ylabel(cp1.label if r == 0 else cp2.label)
                    ax.set_xlabel(cp2.label if r == 1 else cp1.label)

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
        fig.suptitle(f"{cp2.label} ↔ {cp1.label} similarity", fontsize=16)

        plt.savefig(path / "plots" / f"sim_{cp2.label}_{cp1.label}.png", dpi=300)
        plt.close(fig)
