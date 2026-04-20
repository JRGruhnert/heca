from abc import abstractmethod
from dataclasses import dataclass, field
from functools import cached_property

from hoopgn.agents.leaf_agent import LeafAgent
from hoopgn.policies.tapas_policy import TapasPolicy
from hoopgn.properties.features.conditions.condition import PropertyCondition


class TapasAgent(LeafAgent):
    @dataclass(kw_only=True)
    class Config(LeafAgent.Config):
        policy: TapasPolicy.Config = field(init=False)
        overrides: list[str] = field(default_factory=list)

        def __post_init__(self):
            self.policy = TapasPolicy.Config(
                label=self.signature.label,
                reversed=len(self.overrides) != 0,
                overrides=set(self.overrides),
            )

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        self.policy = TapasPolicy.from_config(self.cfg.policy)

    @cached_property
    @abstractmethod
    def v1_property_labels(self) -> set[str]:
        return set(self.v1_precons.keys()) | set(self.v1_postcons.keys())

    @cached_property
    @abstractmethod
    def v1_precons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()
        # return self.policy.load_precons()

    @cached_property
    @abstractmethod
    def v1_postcons(self) -> dict[str, PropertyCondition]:
        raise NotImplementedError()
        # return self.policy.load_postcons()
