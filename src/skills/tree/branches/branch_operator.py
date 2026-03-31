from dataclasses import dataclass


@dataclass
class BranchOperatorConfig:
    pass


class BranchOperator:
    def __init__(self, config: BranchOperatorConfig):
        self.config = config
