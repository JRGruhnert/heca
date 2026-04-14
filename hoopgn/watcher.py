from dataclasses import dataclass


@dataclass(kw_only=True)
class WatcherConfig:
    patience: int = 5
    ema_mode: bool = False
    ema_smoothing_factor: float = 0.1
    min_batches: int = 20
    max_batches: int = 100


class Watcher:
    def __init__(self, config: WatcherConfig):
        self.config = config
        self.best_value = -float("inf")
        self.counter = 0
        self.ema = None

    def update(self, metric: float, current_batch: int) -> bool:
        """
        Determines if training should stop and whether the metric is a new high.
        Returns:
            should_stop (bool): Whether early stopping should trigger.
            is_new_high (bool): Whether the current metric is a new high.
        """
        if self.config.ema_mode:
            should_stop = self._ema_check(metric)
        else:
            should_stop = self._metric_check(metric)

        # Enforce min and max batch constraints
        if current_batch < self.config.min_batches:
            return False
        elif current_batch >= self.config.max_batches:
            return True
        else:
            return should_stop

    def _ema_check(self, metric: float) -> bool:
        # Update EMA
        if self.ema is None:
            self.ema = metric  # Initialize EMA with the first metric value
        else:
            self.ema = (
                self.config.ema_smoothing_factor * metric
                + (1 - self.config.ema_smoothing_factor) * self.ema
            )

        return self._metric_check(self.ema)

    def _metric_check(self, metric: float) -> bool:
        if metric > self.best_value:
            self.best_value = metric
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.config.patience
