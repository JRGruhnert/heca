from dataclasses import dataclass
import torch
from torch import nn
from torch.nn.utils.clip_grad import clip_grad_norm_
from hoopgn.agents.branch_agent import BranchAgent
from hoopgn.hardware import device
from hoopgn.buffer import Buffer, BufferConfig
from hoopgn.observation.td_properties import TDProperties
from hoopgn.networks.mp_final import MPNetwork
from hoopgn import logger
from thop import profile

from hoopgn.agents.agent import Agent


class HoopgnAgent(BranchAgent):
    @dataclass(kw_only=True)
    class Config(BranchAgent.Config):
        network: MPNetwork.Config
        buffer: BufferConfig
        saving_path: str = "results/checkpoints/hoopgn/"
        saving_freq: int = 5  # Saving frequence of trained model

        # PPO Hyperparameters
        mini_batch_size: int = 64  # 64 # How many steps to use in each mini-batch
        learning_epochs: int = 5  # How many passes over the collected batch per update
        lr_annealing: bool = False
        learning_rate: float = 0.0003  # Step size for actor optimizer
        gamma: float = 0.99  # How much future rewards are worth today
        gae_lambda: float = 0.95  # Bias/variance trade‑off in advantage estimation
        eps_clip: float = 0.2  # How far the new policy is allowed to move from the old
        entropy_coef: float = (
            0.02  # Weight on the entropy bonus to encourage exploration
        )
        critic_coef: float = (
            0.5  # Weight on the critic (value) loss vs. the policy loss
        )
        max_grad_norm: float = 0.5  # Threshold for clipping gradient norms
        target_kl: float | None = 0.02  # (Optional) early stopping if KL
        clip_value_loss: bool = True

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.cfg = cfg
        ### Initialize the agent
        self.buffer = Buffer(cfg.buffer)
        self.mse_loss = nn.MSELoss()
        self.policy_new: MPNetwork = MPNetwork.from_config(cfg.network)
        self.policy_old: MPNetwork = MPNetwork.from_config(cfg.network)
        self.optimizer = torch.optim.AdamW(
            self.policy_new.parameters(),
            lr=self.cfg.learning_rate,
        )

    def act(self, obs: TDProperties, goal: TDProperties) -> Agent:
        return self.policy_old.predict(obs, goal)

    def explain(
        self, current: TDProperties, goal: TDProperties
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.policy_old.explain(current, goal)

    def feedback(self, reward: float, success: bool, terminal: bool) -> bool:
        return self.buffer.feedback(reward, success, terminal)

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[torch.Tensor],
        is_terminals: list[bool],
    ):
        advantages = []
        gae = 0
        values = values + [torch.tensor(0.0, device=device)]  # add dummy for V(s_{T+1})
        # TODO: What is that dummy value for? Should it be 0 or something else? Does it matter? Maybe we can just not add it and handle the edge case in the loop?
        for step in reversed(range(len(rewards))):
            terminal = float(is_terminals[step])
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
        assert (
            adv_tensor.shape[0] == self.buffer.config.size
        ), "Advantages shape mismatch"
        assert rtn_tensor.shape[0] == self.buffer.config.size, "Returns shape mismatch"

        return adv_tensor.to(device), rtn_tensor.to(device)

    def learn(self) -> bool:
        assert self.buffer.full, "Buffer must be full before learning!"

        if self.buffer.new_highscore:
            self.save(highscore=True)

        # Main PPO update loop
        self.mini_batch_loop()

        if self.cfg.lr_annealing:
            lr = self.cfg.learning_rate * (1.0 - self.buffer.progress)
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr

        self.update_network()

        if self.buffer.epoch % self.cfg.saving_freq == 0 and self.buffer.epoch != 0:
            self.save(highscore=False)

        self.buffer.save(self.cfg.signature.label)

        return self.buffer.reached_max_batches

    def update_network(self):
        self.policy_old.load_state_dict(self.policy_new.state_dict())

    def mini_batch_loop(
        self,
    ):
        adv, rtn = self._compute_gae(
            self.buffer.rewards,
            self.buffer.values,
            self.buffer.terminals,
        )

        old_obs = self.buffer.current
        old_goal = self.buffer.goal
        old_actions = (
            torch.stack(self.buffer.actions, dim=0).detach().to(device).squeeze(-1)
        )
        old_logprobs = (
            torch.stack(self.buffer.logprobs, dim=0).detach().to(device).squeeze(-1)
        )
        ### Training loop for network
        kl_divergence_stop = False
        for epoch in range(self.cfg.learning_epochs):
            # Shuffle indices for minibatch
            indices = torch.randperm(self.buffer.config.size)

            for start in range(
                0,
                self.buffer.config.size,
                self.cfg.mini_batch_size,
            ):
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
                logprobs, state_values, dist_new = self.policy_new.evaluate(
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
                        torch.stack([self.buffer.values[i] for i in mb_idx_list])
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
                clip_grad_norm_(self.policy_new.parameters(), self.cfg.max_grad_norm)
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

    def save(self, highscore: bool):
        if highscore:
            tag = "highscore"
        else:
            tag = "checkpoint"

        ckpt_path = self.cfg.saving_path + tag + "_epoch{}.pt".format(self.buffer.epoch)

        logger.info(f"Saving weights to: {ckpt_path} at epoch {self.buffer.epoch}")
        torch.save(
            {
                "model_state": self.policy_old.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "epoch": self.buffer.epoch,
            },
            ckpt_path,
        )

    def metadata(self) -> dict:
        return {
            "mini_batch_size": self.cfg.mini_batch_size,
            "learning_epochs": self.cfg.learning_epochs,
            "lr_annealing": self.cfg.lr_annealing,
            "learning_rate": self.cfg.learning_rate,
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
        """Measure FLOPs for a single forward pass through the policy network."""
        with torch.no_grad():
            obs_batch = [obs]
            goal_batch = [goal]
            result = profile(
                self.policy_new, inputs=(obs_batch, goal_batch), verbose=False
            )
            return int(result[0]), int(result[1])
