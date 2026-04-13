from dataclasses import dataclass
from abc import ABC, abstractmethod
import os

from matplotlib import pyplot as plt
from hoopgn import logger


@dataclass
class StyleConfig:
    grey: str = "grey"
    black: str = "black"
    white: str = "white"
    solved: str = "tab:green"
    unsolved: str = "tab:red"
    true: str = "tab:yellow"
    false: str = "black"
    entity: dict[str, str] = {
        "ee": "black",
        "block_red": "tab:red",
        "block_blue": "tab:blue",
        "block_pink": "tab:pink",
        "slide": "black",
        "drawer": "black",
        "button": "black",
        "led": "black",
        "lightbulb": "black",
    }


@dataclass
class PlotConfig:
    title: str
    name: str
    subdir: str
    rootdir: str = "plots"
    style: StyleConfig = StyleConfig()


class Plot(ABC):
    def __init__(self, config: PlotConfig):
        self.config = config

    @abstractmethod
    def __call__(self):
        raise NotImplementedError()

    def save(self):
        logger.info(f"Saving Plot: {self.config.name}")
        save_dir = os.path.join(self.config.rootdir, self.config.subdir)
        os.makedirs(save_dir, exist_ok=True)
        plot_path = os.path.join(save_dir, f"{self.config.name}.png")
        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
