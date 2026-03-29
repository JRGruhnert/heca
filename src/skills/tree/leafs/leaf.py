from dataclasses import dataclass
from functools import cached_property
import torch
from src.observation.observation import StateValueDict
from src.skills.tree.leafs.loader import LeafLoader, LeafLoaderConfig
from src.skills.tree.leafs.operator import OperatorConfig
from src.states.logic.condition import Condition
from src.states.logic.distance import DistanceConfig


@dataclass
class LeafConfig:
    label: str
    id: int
    distance: DistanceConfig
    loader: LeafLoaderConfig
    operator: OperatorConfig


class Leaf:
    def __init__(
        self,
        config: LeafConfig,
    ):
        self.config = config
        self.loader = LeafLoader(config.loader)
        self.operator = self.loader.load_operator(config.operator)
        self.precons: dict[str, Condition] = self.loader.load_precons()
        self.postcons: dict[str, Condition] = self.loader.load_postcons()

        self.goal: StateValueDict | None = None

    def reset(self, *args, **kwargs):
        """Prepare the skill for execution. Before each use."""
        self.operator.reset(*args, **kwargs)

    def act(self, current: StateValueDict, goal: StateValueDict) -> torch.Tensor:
        """Get the next action for the skill."""
        return self.operator.act(current, goal)

    @cached_property
    def demo_precons(self) -> dict[str, torch.Tensor]:
        return self.loader.load_demo_precons()

    @cached_property
    def demo_postcons(self) -> dict[str, torch.Tensor]:
        return self.loader.load_demo_postcons()

    def distances(
        self,
        obs: StateValueDict,
        pad: bool = False,
        sparse: bool = False,
    ) -> torch.Tensor:
        task_features: list[torch.Tensor] = []
        for state in obs.keys():  # type: ignore
            cnd = self.precons.get(str(state), None)
            if cnd:
                value = cnd(obs[state])
                value = torch.tensor([value, 0.0]) if pad else torch.tensor([value])
            else:
                nv = -1.0 if sparse else 0.0  # For Identification in filtering
                value = torch.tensor([nv, 1.0]) if pad else torch.tensor([nv])
            task_features.append(value)
        return torch.stack(task_features, dim=0)
