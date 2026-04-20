from dataclasses import dataclass, field

from hoopgn.agents.agent import Agent
from hoopgn.policies.tapas_policy import TapasPolicy


@dataclass(kw_only=True)
class TapasConfig(Agent.Config):

    operator: TapasPolicy.Config = field(init=False)
    overrides: list[str] = field(default_factory=list)

    def __post_init__(self):
        self.operator = TapasPolicy.Config(
            label=self.ident.label,
            reversed=len(self.ident.overrides) != 0,
            overrides=set(self.ident.overrides),
        )
