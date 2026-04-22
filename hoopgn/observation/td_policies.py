from typing import cast

from tensordict import TensorDict


class TDPolicies(TensorDict):
    def get(self, label: str) -> TensorDict:
        if label in self:
            return self[label]
        assert False, f"Policy with label '{label}' not found."

    def add(self, label: str, policy: TensorDict):
        self[label] = policy

    def __getitem__(self, key: str) -> TensorDict:
        return cast(TensorDict, super().__getitem__(key))
