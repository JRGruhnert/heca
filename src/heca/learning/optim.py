import torch


class ClientAdamW(torch.optim.Optimizer):
    """The client-side AdamW of FedAdamW (arXiv:2510.27486).

    The step itself is ``torch.optim.AdamW``'s, operation for operation; what
    differs is where the moments and their step counter come from at a round
    boundary. The paper's own implementation (github.com/junkangLiu0/FedAdamW,
    ``update_FedAdamW``) builds a fresh ``torch.optim.AdamW`` every round and
    injects the state by hand:

        optimizer.state[p]["step"] = <global step count>   # never reset
        optimizer.state[p]["exp_avg"] = zeros_like(p)      # m = 0
        optimizer.state[p]["exp_avg_sq"] = v_bar[name]     # the server's average

    so a round starts from a zero first moment and the server's second moment, and
    one global count ``t`` debiases both. That is what this class does; leaving
    ``start_round``'s seed out is the plain reset, where the counter goes back to
    zero with the moments - the same as constructing a fresh optimizer, which is
    what the paper's local AdamW (Algorithm 1) and their FedAvg baseline do.

    One caveat, kept because it is what the released code does: Algorithm 2 line 11
    asks for the first moment to be corrected by the round's own index ``k``
    (``1 - beta1**k``). The code hands the global ``t`` to both corrections, so a
    freshly zeroed ``m`` is divided by ``1 - beta1**t``, which is all but 1 once
    training is under way. The first ~30 local steps of every round are therefore
    up to 10x smaller than the ones after them (``m`` only reaches its running
    value after ``1 / (1 - beta1)`` steps). The alpha sweep in the paper was run
    with that schedule, so this follows it rather than the pseudocode.
    """

    def __init__(
        self,
        network,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        params = list(network.named_parameters())
        super().__init__(
            [p for _, p in params],
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        # v_bar is keyed by parameter name, so the round boundary needs the names
        self._named = dict(params)

    @torch.no_grad()
    def step(self) -> None:
        for group in self.param_groups:
            lr: float = group["lr"]
            beta1, beta2 = group["betas"]
            eps: float = group["eps"]
            decay: float = group["weight_decay"]
            for param in group["params"]:
                if param.grad is None:
                    continue
                state = self.state[param]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)
                    state["step"] = 0
                mean, variance = state["exp_avg"], state["exp_avg_sq"]
                gradient = param.grad
                mean.lerp_(gradient, 1 - beta1)
                variance.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                state["step"] += 1
                step_size = lr / (1 - beta1 ** state["step"])
                correction = (1 - beta2 ** state["step"]) ** 0.5
                param.mul_(1 - lr * decay)
                param.addcdiv_(
                    mean, (variance.sqrt() / correction).add_(eps), value=-step_size
                )

    def start_round(self, second_moment: dict[str, torch.Tensor] | None = None) -> None:
        """Set up the moments at a round boundary: ``m <- 0``, ``v <- v_bar`` or 0."""
        seed = second_moment or {}
        # what a tensor missing from v_bar (no gradient for it this round) inherits
        history = max((s.get("step", 0) for s in self.state.values()), default=0)
        for name, param in self._named.items():
            state = self.state[param]
            state["exp_avg"] = torch.zeros_like(param)
            shared = seed.get(name)
            if shared is None:
                state["exp_avg_sq"] = torch.zeros_like(param)
                state["step"] = 0
            else:
                state["exp_avg_sq"] = (
                    shared.detach().to(device=param.device, dtype=param.dtype).clone()
                )
                # v_bar carries the previous rounds, so t keeps running
                state.setdefault("step", history)
