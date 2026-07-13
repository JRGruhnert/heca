from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from stepmix import StepMix

from heca.conditions.condition import Condition
from heca.conditions.pair import ConPair
from scipy.optimize import minimize


class ConditionAnalyzer:
    def __init__(self, threshold: float):
        self.threshold = threshold

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

    def is_subgoal(self, values: dict[str, tuple[float, np.ndarray]]) -> bool:
        if len(values) == 0:
            return False  # No matching keys so no option at all
        for score, _ in values.values():
            if score < self.threshold:
                return False
        return True

    def make_subgoal(
        self, post: Condition, pre: Condition
    ) -> dict[str, tuple[float, np.ndarray]]:
        values = {}
        for key in post.elabels.intersection(pre.elabels):
            score = self.calculate_score(post, pre, key)
            value = self.best_sample(post, pre, key)
            values[key] = (score, value)
        return values

    def best_sample(self, post: Condition, pre: Condition, key: str):

        m1 = post.models[key]
        m2 = pre.models[key]
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

        K1, S1 = state1.shape
        K2, S2 = state2.shape

        S = max(S1, S2)  # union of state space size

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

                # Categorical part
                p_state1 = state1[i]
                p_state2 = state2[j]

                log_cat = -np.inf
                best_state = None

                for s in range(S):

                    p1_s = p_state1[s] if s < S1 else 0.0
                    p2_s = p_state2[s] if s < S2 else 0.0

                    score_s = p1_s * p2_s

                    if score_s > 0:
                        log_score = np.log(score_s)

                        if log_score > log_cat:
                            log_cat = log_score
                            best_state = s

                # if absolutely no overlap in states
                if best_state is None:
                    log_cat = -1e9
                    best_state = 0  # arbitrary fallback

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

    def calculate_sim_matrix(self, cp1: ConPair, cp2: ConPair, key: str) -> np.ndarray:
        mat = np.zeros((2, 2))
        for i, cp1con in enumerate([cp1.pre, cp1.post]):
            for j, cp2con in enumerate([cp2.pre, cp2.post]):
                if key not in cp1con.model_states or key not in cp2con.model_states:
                    mat[i, j] = np.nan
                elif not cp1con.model_states[key].issubset(cp2con.model_states[key]):
                    continue
                else:
                    mat[i, j] = self.kl_variational_stepmix(
                        cp2con.models[key], cp1con.models[key]
                    )
        return mat

    def calculate_score(self, a: Condition, b: Condition, key: str) -> float:
        raw = a.models[key].score(b.samples[key])
        delta = raw - a.sample_self_scores[key]
        clipped = np.minimum(delta, 0)  # we just care for negative deltas
        return np.exp(clipped)  # make a score

    def kl_variational_stepmix(self, m1: StepMix, m2: StepMix):
        p1 = m1.get_parameters()
        p2 = m2.get_parameters()

        kl = 0.0
        for i in range(len(p1["weights"])):
            w1 = p1["weights"][i]
            mu1 = p1["measurement"]["pose"]["means"][i]
            var1 = p1["measurement"]["pose"]["covariances"][i]
            cat1 = p1["measurement"]["state"]["pis"][i]  # ← shape (S,)

            # --- CHANGE STARTS HERE ---
            # Instead of tracking best_kl, we now accumulate the weighted sum
            weighted_pairwise_kl = 0.0

            for j in range(len(p2["weights"])):
                vj = p2["weights"][j]  # weight of component j in m2
                mu2 = p2["measurement"]["pose"]["means"][j]
                var2 = p2["measurement"]["pose"]["covariances"][j]
                cat2 = p2["measurement"]["state"]["pis"][j]  # ← shape (S,)

                # Gaussian part (same as before)
                kl_gauss = 0.5 * np.sum(
                    np.log(var2)
                    - np.log(var1)
                    + var1 / var2
                    + (mu1 - mu2) ** 2 / var2
                    - 1
                )

                # Categorical part (same as before)
                cat1_safe = np.clip(cat1, 1e-12, 1)
                cat2_safe = np.clip(cat2, 1e-12, 1)
                kl_cat = np.sum(cat1_safe * (np.log(cat1_safe) - np.log(cat2_safe)))

                # Weight this pairwise KL by the weight of component j in m2
                weighted_pairwise_kl += vj * (kl_gauss + kl_cat)

            # Now weight the entire sum by the weight of component i in m1
            kl += w1 * weighted_pairwise_kl

        return np.exp(-kl)

    def compute_sim(self, cp1: "ConPair", cp2: "ConPair") -> dict[str, np.ndarray]:
        sim_rating = {}
        print(cp1.elabels)
        print(cp2.elabels)
        for el in cp1.elabels.intersection(cp2.elabels):
            forward = self.calculate_sim_matrix(cp1, cp2, el)
            backward = self.calculate_sim_matrix(cp2, cp1, el)
            sim_rating[el] = np.stack((forward, backward), axis=0)

        return sim_rating

    def plot_similarity(
        self,
        sim_rating: dict[str, np.ndarray],
        cp1: "ConPair",
        cp2: "ConPair",
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
                    ax.set_ylabel(cp1.label if r == 0 else cp2.label)
                    ax.set_xlabel(cp2.label if r == 0 else cp1.label)

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
