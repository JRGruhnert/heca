from dataclasses import dataclass


@dataclass(kw_only=True)
class BranchOperatorConfig:
    pass


class BranchOperator:
    def __init__(self, config: BranchOperatorConfig):
        self.config = config
