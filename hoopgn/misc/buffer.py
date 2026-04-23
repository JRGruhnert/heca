from dataclasses import dataclass
import torch
from hoopgn.classes import StoragableClass
from hoopgn.misc import logger
from hoopgn.observation.td_properties import TDProperties
from hoopgn.observation.td_scene import TDScene
from hoopgn.misc.watcher import Watcher, WatcherConfig


class Buffer(StoragableClass):
    @dataclass(kw_only=True)
    class Config:
        watcher: WatcherConfig = WatcherConfig()
        epoch: int = 0

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.watcher = Watcher(cfg.watcher)

        # Epoch tracking for saving
        self.epoch = cfg.epoch
        self.highscore = 0.0
        self.new_highscore = False
        # We store lists of data for each episode, and ensure they are all the same length before saving.
        self.current: list[TDScene] = []
        self.goal: list[TDScene] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.success: list[bool] = []
        self.terminals: list[bool] = []

    @property
    def full(self) -> bool:
        return len(self.actions) >= self.cfg.size

    @property
    def progress(self) -> float:
        return float(self.epoch / self.cfg.watcher.max_batches)

    @property
    def reached_max_batches(self) -> bool:
        return self.epoch >= self.cfg.watcher.max_batches

    def health(self):
        lengths = [
            len(self.current),
            len(self.goal),
            len(self.actions),
            len(self.logprobs),
            len(self.rewards),
            len(self.success),
            len(self.values),
            len(self.terminals),
        ]
        return all(l == lengths[0] for l in lengths)

    def to_disk(self, tag: str):
        assert self.health(), "Buffer lengths are inconsistent!"
        logger.info(f"Saving {tag} buffer for {self.epoch}")
        torch.save(
            {
                "actions": torch.stack(self.actions),
                "logprobs": torch.tensor(self.logprobs),
                "values": torch.tensor(self.values),
                "rewards": torch.tensor(self.rewards),
                "success": torch.tensor(self.success),
                "terminals": torch.tensor(self.terminals),
            },
            self.cfg.save_path + "/" + tag + f"/epoch_data_{self.epoch}.pt",
        )
        self.current.clear()
        self.goal.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.success.clear()
        self.values.clear()
        self.terminals.clear()

    def act_values(
        self,
        current: TDProperties,
        goal: TDProperties,
        action: torch.Tensor,
        action_logprob: torch.Tensor,
        state_val: torch.Tensor,
    ):
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(action)
        self.logprobs.append(action_logprob)
        self.values.append(state_val)

    def act_values_tree(
        self,
        current: TDProperties,
        goal: TDProperties,
        action: int,
    ):
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(torch.tensor(action))
        self.logprobs.append(torch.tensor(0.0))
        self.values.append(torch.tensor(0.0))

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        self.rewards.append(reward)
        self.success.append(success)
        self.terminals.append(terminal)

        if self.watcher.update(self.epoch_sr, self.epoch):
            logger.info(
                f"Early stopping after {self.epoch} epochs because of no improvement."
            )
            return True

        self.epoch += 1
        self.epoch_sr = sum(self.success) / len(self.success)
        if self.highscore < self.epoch_sr:
            self.highscore = self.epoch_sr
            self.new_highscore = True
            logger.info(f"New highscore: {self.highscore:.4f} at epoch {self.epoch}.")

        return len(self.actions) == self.cfg.size

    def metrics(self) -> dict[str, float]:
        assert self.health(), "Buffer lengths are inconsistent!"
        return {
            "stats/mean_episode_reward": sum(self.rewards) / self.cfg.size,
            "stats/mean_episode_length": self.cfg.size
            / max(1, self.terminals.count(True)),
            "stats/highscore": self.highscore,
        }
