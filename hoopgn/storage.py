from dataclasses import dataclass

import os


@dataclass(kw_only=True)
class StorageConfig:
    tag: str = "untagged_run"
    storage_path: str = "data"
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

    def create_directory(self, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    def agent_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.config.results_path + "/" + network_name + "/" + self.config.tag + "/"
        )
        return self.create_directory(directory_path)

    def buffer_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.buffer_path + "/"
        )
        return self.create_directory(directory_path)

    def plots_saving_path(self, network_name: str) -> str:
        directory_path = (
            self.agent_saving_path(network_name) + self.config.plots_path + "/"
        )
        return self.create_directory(directory_path)
