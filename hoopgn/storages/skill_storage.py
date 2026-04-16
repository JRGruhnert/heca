from dataclasses import dataclass
import os

from hoopgn.storages.storage import Storage, StorageConfig


@dataclass(kw_only=True)
class SkillStorageConfig(StorageConfig):
    root_dir: str = "data"
    skill_dir: str = "skills"
    plot_dir: str = "plots"
    training_dir: str = "training"
    plots_dir: str = "plots"
    checkpoint_dir: str | None = None

    def __post_init__(self):
        self.result_path = self.root_dir + "/" + self.skill_dir
        self.plots_path = self.plot_dir
        self.training_path = self.training_dir
        self.plots_path = self.plots_dir
        if self.checkpoint_dir is not None:
            self.checkpoint_path = self.result_path + "/" + self.checkpoint_dir


class SkillStorage(Storage):
    def __init__(
        self,
        cfg: SkillStorageConfig,
    ):
        super().__init__(cfg)
        self.cfg = cfg

        self.ensure_directories()

    def ensure_directories(self):
        if not os.path.exists(self.cfg.result_path):
            os.makedirs(self.cfg.result_path)
        if not os.path.exists(self.cfg.plots_path):
            os.makedirs(self.cfg.plots_path)
        if not os.path.exists(self.cfg.training_path):
            os.makedirs(self.cfg.training_path)
        if self.cfg.checkpoint_path is not None and not os.path.exists(
            self.cfg.checkpoint_path
        ):
            os.makedirs(self.cfg.checkpoint_path)

    def save(self, network_name: str, data: dict):
        self.buffer.save()
        self.agent.save()
        saving_path = self.agent_saving_path(network_name)

    def load(self, network_name: str) -> dict:
        saving_path = self.agent_saving_path(network_name)
        return {}
