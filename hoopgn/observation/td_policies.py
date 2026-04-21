from typing import cast

from tensordict import TensorDict


class TDPolicies(TensorDict):
    def add(self, label: str, policy: TensorDict):
        self[label] = policy

    def __getitem__(self, key: str) -> TensorDict:
        return cast(TensorDict, super().__getitem__(key))
