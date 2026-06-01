from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn.functional as F

from heca.classes.config import Configurable


class ScoreMode(Enum):
    HIGHEST = "highest"
    AVERAGE = "average"
    RAW = "raw"


class CompareMode(Enum):
    COSINE = "cosine"
    CROSS = "cross_correlation"


class SelectionMode(Enum):
    WEIGHTED_VOTE = "weighted_vote"
    MAJORITY_VOTE = "majority_vote"


class EntityStateKNN(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        top_k: int
        score_mode: ScoreMode
        compare_mode: CompareMode
        selection_mode: SelectionMode

    def __init__(self, cfg: Config):
        self.cfg = cfg
        assert (
            self.cfg.compare_mode is CompareMode.COSINE
        ), "Only cosine compare mode is currently implemented."
        assert self.cfg.top_k % 2 == 1, "top_k must be odd."

        self.ref_descs: dict[str, torch.Tensor] = {}
        self.ref_states: dict[str, list[str]] = {}

    def _process_kernel(self, state_desc_kernel: torch.Tensor) -> torch.Tensor:
        state_desc = state_desc_kernel.flatten().unsqueeze(0)  # (1, D)
        state_desc = F.normalize(state_desc, dim=1)
        return state_desc

    def register(
        self,
        entity_label: str,
        state_label: str,
        state_desc_kernel: torch.Tensor,
    ):
        state_desc = self._process_kernel(state_desc_kernel)
        if entity_label not in self.ref_descs:
            self.ref_descs[entity_label] = state_desc
            self.ref_states[entity_label] = [state_label]
        else:
            self.ref_descs[entity_label] = torch.cat(
                [self.ref_descs[entity_label], state_desc],
                dim=0,
            )
            self.ref_states[entity_label].append(state_label)

    def query(
        self,
        entity_label: str,
        state_desc_kernel: torch.Tensor,
    ) -> tuple[str, float]:
        state_desc = self._process_kernel(state_desc_kernel)
        ref_descriptors = self.ref_descs[entity_label]
        state_labels = self.ref_states[entity_label]
        scores = torch.matmul(ref_descriptors, state_desc.T).squeeze(1)
        top_scores, top_indices = torch.topk(scores, k=self.cfg.top_k)

        state_scores = {state: [] for state in state_labels}
        for k_idx, k_score in zip(top_indices.tolist(), top_scores.tolist()):
            state_label = state_labels[k_idx]
            state_scores[state_label].append(float(k_score))

        if self.cfg.selection_mode == SelectionMode.WEIGHTED_VOTE:
            votes = {
                state_label: sum(state_scores[state_label])
                for state_label in state_labels
            }
        elif self.cfg.selection_mode == SelectionMode.MAJORITY_VOTE:
            votes = {
                state_label: len(state_scores[state_label])
                for state_label in state_labels
            }
        else:
            raise ValueError(f"Unsupported selection mode: {self.cfg.selection_mode}")

        prediction = max(votes.items(), key=lambda x: x[1])[0]

        if self.cfg.score_mode == ScoreMode.HIGHEST:
            confidence = max(state_scores[prediction])
        elif self.cfg.score_mode == ScoreMode.AVERAGE:
            confidence = sum(state_scores[prediction]) / len(state_scores[prediction])
        elif self.cfg.score_mode == ScoreMode.RAW:
            confidence = sum(state_scores[prediction])
        return prediction, confidence
