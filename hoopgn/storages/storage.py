from dataclasses import dataclass

import os


@dataclass(kw_only=True)
class StorageConfig:
    tag: str
    root_path: str
    results_path: str
    buffer_path: str
    plots_path: str


@dataclass(kw_only=True)
class ResultsStorageConfig:
    root_path: str = "data"
    results_path: str = "results"
    buffer_path: str = "logs"
    plots_path: str = "plots"
    checkpoint_path: str | None = None


class Storage:
    def __init__(
        self,
        config: StorageConfig,
    ):
        self.config = config

    def ensure_directories(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def agent_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.config.results_path + "/" + network_name + "/" + self.config.tag + "/"
        )
        return self.ensure_directories(directory_path)

    def buffer_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.buffer_path + "/"
        )
        return self.ensure_directories(directory_path)

    def plots_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.plots_path + "/"
        )
        return self.ensure_directories(directory_path)
