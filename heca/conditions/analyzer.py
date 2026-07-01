from pathlib import Path
import numpy as np
from stepmix import StepMix

from heca.conditions.pair import ConditionPair


class ConditionAnalyzer:
    def __init__(self):
        pass

    def calculate_sim_matrix(
        self,
        cp1: "ConditionPair",
        cp2: "ConditionPair",
        el: str,
    ) -> np.ndarray:
        mat = np.zeros((2, 2))
        for i, cp1con in enumerate([cp1.pre, cp1.post]):
            for j, cp2con in enumerate([cp2.pre, cp2.post]):
                if el not in cp1con.model_states or el not in cp2con.model_states:
                    mat[i, j] = 1.0
                    continue

                if not cp1con.model_states[el].issubset(cp2con.model_states[el]):
                    mat[i, j] = 0.0
                    continue

                raw = cp2con.model[el].score(cp1con.samples[el])
                delta = raw - cp1con.sample_self_scores[el]
                delta = np.clip(delta, -50, 0)  # Overflow pevention
                mat[i, j] = np.exp(delta)
        return mat

    def analyze(
        self,
        cp1: "ConditionPair",
        cp2: "ConditionPair",
        plot_path: Path | None = None,
    ):
        results = {}
        for el in cp1.elabels | cp2.elabels:
            if el not in ["ee", "ee_target"]:
                forward = self.calculate_sim_matrix(cp1, cp2, el)
                backward = self.calculate_sim_matrix(cp2, cp1, el)

                results[el] = {
                    "forward": forward,
                    "backward": backward,
                }

        if plot_path is not None:
            self._plot_similarity(results, cp1.label, cp2.label, plot_path)

        return results

    def _plot_similarity(
        self,
        analyze: dict[str, dict[str, np.ndarray]],
        xl: str,
        yl: str,
        path: Path,
    ):
        import matplotlib.pyplot as plt

        entities = list(analyze.keys())
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

        for c, entity in enumerate(entities):
            for r, direction in enumerate(("forward", "backward")):
                ax = axes[r, c]
                mat = analyze[entity][direction]

                im = ax.imshow(
                    mat,
                    cmap="viridis",
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
                    ax.set_ylabel(xl if direction == "forward" else yl)
                    ax.set_xlabel(yl if direction == "forward" else xl)

                # annotate values
                for i in range(2):
                    for j in range(2):
                        ax.text(
                            j,
                            i,
                            f"{mat[i, j]:.2f}",
                            ha="center",
                            va="center",
                            color="red",
                            fontsize=9,
                        )

        fig.colorbar(im, ax=axes, label="Similarity", shrink=0.8)
        fig.suptitle(f"{yl} ↔ {xl} similarity", fontsize=16)

        plt.savefig(path / f"sim_{yl}_{xl}.png", dpi=300)
        plt.close(fig)
