from pathlib import Path

import numpy as np

from heca.conditions.condition import Condition

# Condition saves raw datapoints per entity


class ConPair:
    def __init__(
        self,
        label: str,
        pre: Condition,
        post: Condition,
    ):
        self.label = label
        self.pre = pre
        self.post = post

    @property
    def elabels(self) -> set[str]:
        # Convienience. elbales should never be different
        assert self.pre.elabels == self.post.elabels
        return self.pre.elabels

    @classmethod
    def merge(cls, label: str, a: "ConPair", b: "ConPair", n_samples: int) -> "ConPair":
        pre_max, post_max = cls.make_max_components(a, b)
        pre_data = cls._merge_data(a.pre.data_raw, b.pre.data_raw)
        post_data = cls._merge_data(a.post.data_raw, b.post.data_raw)
        pre = Condition("pre", pre_data, pre_max, n_samples)
        post = Condition("post", post_data, post_max, n_samples)
        return cls(label, pre, post)

    @classmethod
    def make_max_components(cls, a: "ConPair", b: "ConPair") -> tuple[int, int]:
        pre_max = a.pre._max_components + b.pre._max_components
        post_max = a.post._max_components + b.post._max_components
        return pre_max, post_max

    @classmethod
    def _merge_data(
        cls,
        d1: dict[str, np.ndarray],
        d2: dict[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        result = d1.copy()
        for k, v in d2.items():
            result[k] = np.concatenate((result[k], v), axis=0) if k in result else v
        return result

    def plot(self, path: Path):
        plot_path = path / "plots"
        plot_path.mkdir(parents=True, exist_ok=True)
        self.pre.plot(plot_path)
        self.post.plot(plot_path)
