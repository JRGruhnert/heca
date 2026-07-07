from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from heca.agents.agent import Agent


class GraphNode(ABC):
    x: torch.Tensor
    src: str
    dst: str
    node: str

    @property
    @abstractmethod
    def key(self) -> str:
        raise NotImplementedError


class PosNode(GraphNode):
    x: torch.Tensor
    node: str = "pos"

    @property
    def key(self) -> str:
        raise NotImplementedError


class RotNode(GraphNode):
    x: torch.Tensor
    node: str = "rot"

    @property
    def key(self) -> str:
        raise NotImplementedError


class SteNode(GraphNode):
    x: torch.Tensor
    node: str = "ste"

    @property
    def key(self) -> str:
        raise NotImplementedError


class EntityNode(GraphNode):
    tag: str
    x: Optional[torch.Tensor] = None
    node: str = "entity"

    @property
    def key(self) -> str:
        return self.dst + self.tag

    @classmethod
    def make_key(cls, etag: str, tag: str) -> str:
        return etag + tag


class PosCompNode(GraphNode):
    node: str = "posc"

    @property
    def key(self) -> str:
        raise NotImplementedError


class RotCompNode(GraphNode):
    x: torch.Tensor
    node: str = "rotc"

    @property
    def key(self) -> str:
        raise NotImplementedError


class SteCompNode(GraphNode):
    x: torch.Tensor
    node: str = "stec"

    @property
    def key(self) -> str:
        raise NotImplementedError


class CompNode(GraphNode):
    idx: int
    x: Optional[torch.Tensor] = None
    node: str = "comp"

    @property
    def key(self) -> str:
        return self.dst + f"{self.idx}"


class PreMixNode(GraphNode):
    etag: str
    x: Optional[torch.Tensor] = None
    node: str = "premix"

    @property
    def key(self) -> str:
        return self.dst + self.etag + self.node


class PostMixNode(GraphNode):
    etag: str
    x: Optional[torch.Tensor] = None
    node: str = "postmix"

    @property
    def key(self) -> str:
        return self.dst + self.node


class OptionNode(GraphNode):
    agent: Agent.Config
    x: Optional[torch.Tensor] = None
    node: str = "option"
