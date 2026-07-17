import numpy as np


class RewardNormalizer:
    def __init__(self, clip_range=10.0, eps=1e-8):
        self.clip_range = clip_range
        self.eps = eps
        self.count = 0
        self.mean = 0.0
        self.var = 1.0  # Starts at 1 to avoid division by zero early on

    def update(self, reward: float) -> float:
        """Update running statistics and return the normalized reward."""
        # 1. Update running mean and variance (Welford's online algorithm)
        self.count += 1
        delta = reward - self.mean
        self.mean += delta / self.count
        delta2 = reward - self.mean
        self.var += delta * delta2

        # 2. Calculate standard deviation (only when count > 1)
        if self.count > 1:
            std = np.sqrt(self.var / (self.count - 1)) + self.eps
        else:
            std = 1.0  # Fallback for the very first step

        # 3. Normalize and clip
        normalized = (reward - self.mean) / std
        return np.clip(normalized, -self.clip_range, self.clip_range)

    def save_stats(self):
        """Save these values to disk for evaluation."""
        return {"count": self.count, "mean": self.mean, "var": self.var}

    def load_stats(self, count, mean, var):
        """Load saved stats for evaluation so normalization is consistent."""
        self.count = count
        self.mean = mean
        self.var = var
