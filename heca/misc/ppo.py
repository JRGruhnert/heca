from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from heca.misc.classes import ConfigClass
from heca.misc.hardware import device
from heca.heca_gnn.hecagn import HecaGN
from heca.misc import logger
from thop import profile

from heca.misc.td import TDScene, TDProperties


@dataclass(kw_only=True)
class AnnealingConfig:
    lr_annealing: bool = False
    learning_rate: float = 0.0003
    max_epochs: int = 1000


class PPO(ConfigClass):
    @dataclass(kw_only=True)
    class Config(ConfigClass.Config):
        # PPO Hyperparameters
        batch_size: int = 2048
        mini_batch_size: int = 64
        learning_epochs: int = 5
        lr: float = 0.0003
        lr_annealing: bool = False
        gamma: float = 0.99
        gae_lambda: float = 0.95
        eps_clip: float = 0.2
        entropy_coef: float = 0.02
        critic_coef: float = 0.5
        max_grad_norm: float = 0.5
        target_kl: float | None = 0.02
        clip_value_loss: bool = True

    def __init__(self, cfg: Config, hoop: HecaGN):
        self.cfg = cfg

        self.hoop = hoop
        self.mse_loss = nn.MSELoss()
        self.optimizer = torch.optim.AdamW(
            self.hoop.parameters(),
            lr=self.cfg.lr,
        )
        self.current: list[TDScene] = []
        self.goal: list[TDScene] = []
        self.actions: list[torch.Tensor] = []
        self.logprobs: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []
        self.rewards: list[float] = []
        self.success: list[bool] = []
        self.terminals: list[bool] = []

        # Statistics
        self.highscore = float("-inf")

    def learn(self, progress: float) -> tuple[dict, bool]:
        # Main PPO update loop
        self.mini_batch_loop()

        # Update learning rate with linear annealing
        if self.cfg.lr_annealing:
            assert (
                progress is not None
            ), "Progress must be provided for learning rate annealing"
            lr = self.cfg.lr * (1.0 - progress)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        return self.hoop.state_dict(), self.flush_and_rate()

    def mini_batch_loop(self):
        adv, rtn = self._compute_gae(
            self.rewards,
            self.values,
            self.terminals,
        )

        old_obs = self.current
        old_goal = self.goal
        old_actions = torch.stack(self.actions, dim=0).detach().to(device).squeeze(-1)
        old_logprobs = torch.stack(self.logprobs, dim=0).detach().to(device).squeeze(-1)
        ### Training loop for network
        kl_divergence_stop = False
        for epoch in range(self.cfg.learning_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.cfg.batch_size)
            for start in range(0, self.cfg.batch_size, self.cfg.mini_batch_size):
                end = start + self.cfg.mini_batch_size
                mb_idx = indices[start:end]
                mb_idx_list = mb_idx.tolist()  # turn Tensor → Python list of ints
                # Decided to save observations as objects instead of tensors
                # Makes it easier to convert it based on network later on
                mb_obs = [old_obs[i] for i in mb_idx_list]
                mb_goal = [old_goal[i] for i in mb_idx_list]
                mb_actions = old_actions[mb_idx]
                mb_logprobs = old_logprobs[mb_idx]
                mb_adv = adv[mb_idx]
                mb_rtn = rtn[mb_idx]

                # Normalize advantages per minibatch
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # Evaluate policy
                logprobs, state_values, dist_new = self.hoop.evaluate(
                    mb_obs, mb_goal, mb_actions
                )

                assert logprobs.shape == mb_logprobs.shape, "Logprobs shape mismatch"
                assert (
                    state_values.shape == mb_rtn.shape
                ), "Value prediction shape mismatch"

                state_values = torch.squeeze(state_values)

                # Ratios
                ratios = torch.exp(logprobs - mb_logprobs.detach().to(device))

                # Surrogate loss
                surr1 = ratios * mb_adv
                surr2 = (
                    torch.clamp(
                        ratios,
                        1 - self.cfg.eps_clip,
                        1 + self.cfg.eps_clip,
                    )
                    * mb_adv
                )

                # Value loss (with optional clipping)
                if self.cfg.clip_value_loss:
                    mb_old_values = (
                        torch.stack([self.values[i] for i in mb_idx_list])
                        .squeeze()
                        .to(device)
                    )
                    values_pred = mb_old_values + torch.clamp(
                        state_values - mb_old_values,
                        -self.cfg.eps_clip,
                        self.cfg.eps_clip,
                    )
                    value_loss = self.mse_loss(values_pred, mb_rtn)
                else:
                    value_loss = self.mse_loss(state_values, mb_rtn)

                # Calculate loss
                loss: torch.Tensor = (
                    -torch.min(surr1, surr2)
                    + self.cfg.critic_coef * value_loss
                    - self.cfg.entropy_coef * dist_new.entropy().mean()
                )

                ### Update gradients on mini-batch
                self.optimizer.zero_grad()
                loss.mean().backward()
                clip_grad_norm_(self.hoop.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                # Collect metrics (will keep last minibatch values)
                with torch.no_grad():
                    log_ratio = logprobs - mb_logprobs
                    approx_kl = torch.mean((torch.exp(log_ratio) - 1) - log_ratio)
                    clip_fraction = (
                        ((ratios - 1).abs() > self.cfg.eps_clip).float().mean()
                    )
                    entropy = dist_new.entropy().mean()
                    policy_loss = (-torch.min(surr1, surr2)).mean()

                    # self._metrics.update(
                    #    {
                    #        "ppo/policy_loss": policy_loss.item(),
                    #        "ppo/value_loss": value_loss.item(),
                    #        "ppo/entropy": entropy.item(),
                    #        "ppo/approx_kl": approx_kl.item(),
                    #        "ppo/clip_fraction": clip_fraction.item(),
                    #        "ppo/total_loss": loss.mean().item(),
                    #    }
                    # )
                    # Optional KL early stopping
                    if self.cfg.target_kl is not None:
                        if approx_kl > self.cfg.target_kl:
                            kl_divergence_stop = True
                            break  # break minibatch loop
            if kl_divergence_stop:
                break

    def _compute_gae(
        self, rewards: list[float], values: list[torch.Tensor], terminals: list[bool]
    ):
        assert self.hoop is not None, "Policy must be set before learning"
        assert self.optimizer is not None, "Optimizer must be set before learning"

        advantages = []
        gae = 0
        values = values + [torch.tensor(0.0, device=device)]  # add dummy for V(s_{T+1})
        # TODO: What is that dummy value for? Should it be 0 or something else? Does it matter? Maybe we can just not add it and handle the edge case in the loop?
        for step in reversed(range(len(rewards))):
            terminal = float(terminals[step])
            delta = (
                rewards[step]
                + self.cfg.gamma * values[step + 1] * (1 - terminal)
                - values[step]
            )
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * (1 - terminal) * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values[:-1])]
        adv_tensor = torch.tensor(advantages, dtype=torch.float32)
        rtn_tensor = torch.tensor(returns, dtype=torch.float32)
        assert adv_tensor.shape[0] == rtn_tensor.shape[0], "Advantages shape mismatch"
        return adv_tensor.to(device), rtn_tensor.to(device)

    def metadata(self) -> dict:
        return {
            "mini_batch_size": self.cfg.mini_batch_size,
            "learning_epochs": self.cfg.learning_epochs,
            "lr_annealing": self.cfg.lr_annealing,
            "learning_rate": self.cfg.lr,
            "gamma": self.cfg.gamma,
            "gae_lambda": self.cfg.gae_lambda,
            "eps_clip": self.cfg.eps_clip,
            "entropy_coef": self.cfg.entropy_coef,
            "critic_coef": self.cfg.critic_coef,
            "max_grad_norm": self.cfg.max_grad_norm,
            "clip_value_loss": self.cfg.clip_value_loss,
            **(
                {"target_kl": self.cfg.target_kl}
                if self.cfg.target_kl is not None
                else {}
            ),
        }

    def measure_flops(self, obs: TDProperties, goal: TDProperties) -> tuple[int, int]:
        assert self.hoop is not None, "Hoop must be set before measuring FLOPs"
        with torch.no_grad():
            obs_batch = [obs]
            goal_batch = [goal]
            result = profile(self.hoop, inputs=(obs_batch, goal_batch), verbose=False)
            return int(result[0]), int(result[1])

    def flush_and_rate(self):
        current = len(self.success) / len(self.terminals)
        self.current.clear()
        self.goal.clear()
        self.actions.clear()
        self.logprobs.clear()
        self.rewards.clear()
        self.success.clear()
        self.values.clear()
        self.terminals.clear()
        if current > self.highscore:
            self.highscore = current
            return True
        return False

    def to_disk(self, tag: str):
        assert self.optimizer is not None, "Optimizer must be set before saving"
        logger.info(f"Saving {tag} buffer")
        torch.save(
            {
                "actions": torch.stack(self.actions),
                "logprobs": torch.tensor(self.logprobs),
                "values": torch.tensor(self.values),
                "rewards": torch.tensor(self.rewards),
                "success": torch.tensor(self.success),
                "terminals": torch.tensor(self.terminals),
                "optimizer_state": self.optimizer.state_dict(),
            },
            # TODO: storage path
            "data/" + tag + f"/epoch_data_{0}.pt",
        )

    def step(
        self,
        current: TDProperties,
        goal: TDProperties,
        action: torch.Tensor,
        logprob: torch.Tensor,
        value: torch.Tensor,
        reward: float,
        success: bool,
        terminal: bool,
    ) -> bool:
        self.current.append(current)
        self.goal.append(goal)
        self.actions.append(action)
        self.logprobs.append(logprob)
        self.values.append(value)
        self.rewards.append(reward)
        self.success.append(success)
        self.terminals.append(terminal)
        return len(self.current) >= self.cfg.batch_size
