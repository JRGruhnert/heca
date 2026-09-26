"""The client-side AdamW of FedAdamW (arXiv:2510.27486).

The step is ``torch.optim.AdamW``'s; what a round boundary changes is where the
moments and their step counter come from. The paper's released implementation
(github.com/junkangLiu0/FedAdamW) injects ``step = <global count>``,
``exp_avg = 0`` and ``exp_avg_sq = v_bar`` into a fresh optimizer every round, so
one global count ``t`` debiases both moments, ``m`` starts from zero and ``v`` from
the server's average. Without a seed everything starts from zero, i.e. a plain
fresh optimizer.
"""

import io
from types import SimpleNamespace

import torch

from heca.learning.fppo import FPPO
from heca.learning.optim import ClientAdamW

LR = 3e-4
BETA1 = 0.9
BETA2 = 0.999


class Tiny(torch.nn.Module):
    """A module with named parameters, so the round boundary can be keyed by name."""

    def __init__(self, *weights):
        super().__init__()
        for i, weight in enumerate(weights):
            self.register_parameter(f"w{i}", torch.nn.Parameter(weight.clone()))

    def values(self) -> torch.Tensor:
        return torch.cat([p.detach().reshape(-1) for p in self.parameters()])


def fixture(n: int = 7, steps: int = 200):
    torch.manual_seed(0)
    weights = [torch.randn(n) for _ in range(3)]
    grads = [[torch.randn(n) for _ in range(3)] for _ in range(steps)]
    return weights, grads


def set_grads(net: Tiny, grads) -> None:
    for (_, param), grad in zip(net.named_parameters(), grads):
        param.grad = grad.clone()


def pair(weights):
    left, right = Tiny(*weights), Tiny(*weights)
    return left, ClientAdamW(left, lr=LR, weight_decay=0.01), right


def test_matches_torch_adamw_without_a_round_boundary():
    weights, grads = fixture()
    mine, optim, reference = pair(weights)
    theirs = torch.optim.AdamW(reference.parameters(), lr=LR, weight_decay=0.01)
    for step in grads[:60]:
        set_grads(mine, step)
        set_grads(reference, step)
        optim.step()
        theirs.step()
    assert torch.equal(mine.values(), reference.values())


def test_a_plain_round_is_a_genuinely_fresh_torch_optimizer():
    weights, grads = fixture()
    mine, optim, reference = pair(weights)
    theirs = torch.optim.AdamW(reference.parameters(), lr=LR, weight_decay=0.01)
    for i, step in enumerate(grads[:90]):
        if i and i % 30 == 0:
            optim.start_round()
            for param in reference.parameters():
                state = theirs.state[param]
                state["exp_avg"].zero_()
                state["exp_avg_sq"].zero_()
                state["step"].zero_()
        set_grads(mine, step)
        set_grads(reference, step)
        optim.step()
        theirs.step()
    assert torch.equal(mine.values(), reference.values())


def test_the_step_counter_survives_a_checkpoint():
    weights, grads = fixture()
    left, optim = Tiny(*weights), None
    optim = ClientAdamW(left, lr=LR, weight_decay=0.01)
    right = Tiny(*weights)
    resumed = ClientAdamW(right, lr=LR, weight_decay=0.01)
    for i, step in enumerate(grads[:40]):
        set_grads(left, step)
        set_grads(right, step)
        if i == 20:
            buffer = io.BytesIO()
            torch.save(optim.state_dict(), buffer)
            buffer.seek(0)
            right.load_state_dict(left.state_dict())
            resumed.load_state_dict(torch.load(buffer, weights_only=False))
        optim.step()
        resumed.step()
    assert torch.equal(left.values(), right.values())
    assert "step" in resumed.state[list(right.parameters())[0]]


def probe(seeded: bool, warmup: int = 0, boundary: int = 100):
    """Ratio of the actual step to ``lr``, per local step after a round boundary.

    A constant gradient of 1 and no weight decay: an AdamW whose bias corrections
    match its moments moves by exactly the learning rate. Keyed by the local step
    ``k``, holding the global count ``t`` the step ran with. ``warmup`` counts the
    local steps taken before the boundary.
    """
    net = Tiny(torch.zeros(1))
    optim = ClientAdamW(net, lr=LR, weight_decay=0.0)
    ratios = {}
    for t in range(1 - warmup, 141):
        if t == boundary:
            # the v_bar a client reports for a constant gradient is this running
            # average itself, i.e. ~1.0
            optim.start_round({"w0": torch.ones(1)} if seeded else None)
        before = net.w0.item()
        net.w0.grad = torch.ones(1)
        optim.step()
        if t >= boundary:
            ratios[t - boundary + 1] = (abs(net.w0.item() - before) / LR, t + warmup)
    return ratios


def test_a_plain_boundary_scales_every_step_by_the_counter_it_restarted():
    for ratio, _ in probe(seeded=False).values():
        assert abs(ratio - 1.0) < 1e-4


def test_a_seeded_boundary_keeps_the_global_count_for_both_moments():
    # m is zeroed but still divided by 1 - beta1**t, so it is short by the factor
    # 1 - beta1**k it has not accumulated yet: every round opens with roughly 30
    # damped steps. Only the second moment's correction depends on t.
    for warmup in (0, 5000):
        ratios = probe(seeded=True, warmup=warmup)
        for k, (ratio, t) in ratios.items():
            expected = (1 - BETA1**k) * (1 - BETA2**t) ** 0.5 / (1 - BETA1**t)
            assert abs(ratio - expected) < 2e-3


def make_learner(module: Tiny, alpha: float | None) -> FPPO:
    learner = object.__new__(FPPO)
    learner.cfg = SimpleNamespace(lr=LR, weight_decay=0.0)
    learner.network = module
    learner._sync_keys = {name for name, _ in module.named_parameters()}
    learner._use_fedadamw = alpha is not None
    learner._alpha = float(alpha or 0.0)
    learner._v_bar = {}
    learner._applied = {}
    learner._local_steps = 0
    learner._previous_steps = 0
    learner.optim = learner._make_optimizer()
    learner._global = learner._synced_state()
    learner._global_params = learner._global_snapshot()
    return learner


def rounded(module: Tiny, learner: FPPO, grads, start: int, count: int) -> None:
    for i in range(start, start + count):
        set_grads(module, grads[i])
        learner.optim.step()
        learner._optim_update_hook()


def test_a_round_looks_like_the_paper_describes_it():
    weights, grads = fixture()
    net = Tiny(*weights)
    learner = make_learner(net, alpha=0.5)
    rounded(net, learner, grads, 0, 4)
    assert learner._local_steps == 4

    mean_before = learner._synced_state()
    global_before = dict(learner._global)
    learner._aggregate_distributed()

    # the server's v_bar is the clients' final second moment (line 6 of Algorithm 2)
    for name, param in net.named_parameters():
        assert torch.equal(
            learner._v_bar[name],
            learner.optim.state[param]["exp_avg_sq"].to("cpu", dtype=torch.float32),
        )
    # and the global update is the plain average of the deltas
    for name in learner._applied:
        assert torch.allclose(
            learner._applied[name], mean_before[name] - global_before[name]
        )
        assert torch.allclose(learner._global[name], mean_before[name])

    learner._previous_steps, learner._local_steps = 4, 0
    learner.optim.start_round(learner._v_bar)
    for name, param in net.named_parameters():
        state = learner.optim.state[param]
        assert torch.equal(state["exp_avg_sq"], learner._v_bar[name])
        assert state["exp_avg"].abs().max() == 0
        # v_bar carries the previous rounds, so t keeps counting
        assert state["step"] == 4


def test_the_alignment_moves_the_round_by_alpha_times_the_global_update():
    weights, grads = fixture()
    net = Tiny(*weights)
    aligned = make_learner(net, alpha=0.5)
    rounded(net, aligned, grads, 0, 4)
    aligned._aggregate_distributed()
    aligned._previous_steps, aligned._local_steps = 4, 0
    aligned.optim.start_round(aligned._v_bar)

    start = {name: p.detach().clone() for name, p in net.named_parameters()}
    rounded(net, aligned, grads, 4, 4)
    with_alignment = {
        name: p.detach() - start[name] for name, p in net.named_parameters()
    }

    # the same round without the alignment term (alpha = 0, the A2 ablation)
    plain_net = Tiny(*weights)
    plain = make_learner(plain_net, alpha=0.0)
    rounded(plain_net, plain, grads, 0, 4)
    plain._aggregate_distributed()
    plain._previous_steps, plain._local_steps = 4, 0
    plain.optim.start_round(plain._v_bar)
    assert all(
        torch.equal(p.detach(), start[name]) for name, p in plain_net.named_parameters()
    )
    rounded(plain_net, plain, grads, 4, 4)
    without = {
        name: p.detach() - start[name] for name, p in plain_net.named_parameters()
    }

    for name in aligned._applied:
        # alpha/K per local step over K local steps is alpha over the round
        assert torch.allclose(
            with_alignment[name] - without[name],
            aligned._alpha * aligned._applied[name],
            atol=1e-6,
        )


def test_a_seeded_tensor_the_client_never_stepped_gets_the_global_count():
    # a scene that never touches one of the network's heads still receives that
    # head's v_bar from the other clients, and the estimate it is handed carries
    # their history rather than none of its own
    net = Tiny(torch.zeros(4), torch.zeros(4))
    optim = ClientAdamW(net, lr=LR, weight_decay=0.0)
    for _ in range(3):
        net.w0.grad = torch.ones(4)
        net.w1.grad = None
        optim.step()
    optim.start_round({"w0": torch.ones(4), "w1": torch.full((4,), 2.0)})
    assert optim.state[net.w0]["step"] == 3
    assert optim.state[net.w1]["step"] == 3
    assert torch.equal(optim.state[net.w1]["exp_avg_sq"], torch.full((4,), 2.0))
