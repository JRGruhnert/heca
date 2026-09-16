"""The memory contract: one field in the data, one entry per trunk.

``HecaData.memory`` maps a trunk name to the recurrence that decision was
conditioned on. It is the *only* way memory enters a forward pass:

* the trunk reads ``data.memory[name]``, computes ``h_t = GRU(z_t, h_{t-1})`` in
  the same pass and hands ``h_t`` back with the rows;
* rollout stores that mapping (as plain tensors) into the next transition's data;
* replay swaps in the live mapping for the steps after the first one of a chunk,
  so the GRUs get gradient across the chunk, and restores the recorded mapping
  afterwards.
"""

from typing import NamedTuple

import torch
from torch import nn

from heca.data.entity import Entity
from heca.graphs.data import HecaData, installed_memory
from heca.graphs.roles import ENRole
from heca.heca_gnn.modules.timeline import MemoryBlock
from heca.heca_gnn.trunc import TruncOutput
from heca.learning.learner import _stored
from heca.learning.ppo import score_chunks

DIM = Entity.FEATURE_DIM  # the real memory width


def make_data(**memories: torch.Tensor) -> HecaData:
    data = HecaData()
    data.memory = dict(memories)
    return data


class StubOutput(NamedTuple):
    logits: torch.Tensor
    value: torch.Tensor
    memory: dict[str, torch.Tensor]


class StubNet(nn.Module):
    """Mirrors the trunk contract: read ``data.memory``, return the next memory."""

    def __init__(self, keys: tuple[str, ...] = ("trunk",)):
        super().__init__()
        self.keys = keys
        self.grus = nn.ModuleDict({k: MemoryBlock(DIM) for k in keys})
        self.u = nn.Parameter(torch.randn(3, DIM))  # option embeddings
        self.seen: list[dict[str, torch.Tensor]] = []

    def forward(self, data: HecaData) -> StubOutput:
        self.seen.append(dict(data.memory))
        state = torch.zeros(1, DIM)
        for key in self.keys:
            h = data.memory.get(key)
            state = state + (h if h is not None else torch.zeros(1, DIM))
        logits = self.u.sum(dim=-1, keepdim=True) + state.sum()
        # h_t = GRU(z_t, h_{t-1}) per trunk, exactly as the trunk computes it
        memory = {key: self.grus[key](state, data.memory.get(key)) for key in self.keys}
        return StubOutput(logits.view(1, -1), state.sum().reshape(1), memory)


def test_memory_field_is_per_trunk():
    data = HecaData()
    assert data.memory == {}  # nothing recorded yet

    actor = torch.full((1, DIM), 1.0)
    critic = torch.full((1, DIM), 2.0)
    data.memory = {"actor": actor, "critic": critic}
    assert set(data.memory) == {"actor", "critic"}
    assert data.memory["critic"] is critic
    assert data.to("cpu").memory["actor"] is actor

    # pyg stores attributes by reference (it bypasses property setters), so the
    # mapping is NOT copied on assignment: every caller must hand over a dict it
    # does not mutate afterwards. Pinned here so the contract cannot drift.
    mapping = {"actor": actor}
    data.memory = mapping
    assert data.memory is mapping


def test_installed_memory_restores_recorded_mapping():
    recorded = {"trunk": torch.full((1, DIM), 1.0)}
    live = {"trunk": torch.full((1, DIM), 2.0)}
    data = HecaData()
    data.memory = recorded

    with installed_memory(data, live):
        assert data.memory["trunk"] is live["trunk"]
    assert data.memory["trunk"] is recorded["trunk"]

    # ... also when the block raises (the buffer is replayed again next epoch)
    try:
        with installed_memory(data, live):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert data.memory["trunk"] is recorded["trunk"]


def test_chunk_replay_threads_live_memory_per_trunk():
    torch.manual_seed(2)
    net = StubNet(keys=("actor", "critic"))
    recorded = [
        {
            "actor": torch.full((1, DIM), float(i)),
            "critic": torch.full((1, DIM), float(-i)),
        }
        for i in range(3)
    ]
    data = [make_data(**m) for m in recorded]
    actions = torch.tensor([0, 1, 2])

    logprobs, values, entropies = score_chunks(net, [[0, 1, 2]], data, actions)

    # first step of the chunk: the recorded (detached) memories, per trunk
    assert net.seen[0]["actor"] is data[0].memory["actor"]
    assert not net.seen[0]["actor"].requires_grad
    assert net.seen[0]["critic"] is data[0].memory["critic"]
    # later steps: the live tensors the previous step returned, one per trunk
    for name in ("actor", "critic"):
        assert net.seen[1][name] is not data[1].memory[name]
        assert net.seen[2][name] is not data[2].memory[name]
    assert not torch.allclose(net.seen[2]["actor"], net.seen[2]["critic"])
    # recorded mappings are back in place -> replaying another epoch is safe
    for d, m in zip(data, recorded):
        for name, value in m.items():
            assert torch.equal(d.memory[name], value)

    # the recurrence is unrolled: gradient from the last step reaches both GRUs
    values.sum().backward()
    for name in ("actor", "critic"):
        grads = [p.grad for p in net.grus[name].parameters() if p.grad is not None]
        assert grads, f"no gradient reached the {name} memory cell"
        assert any(bool(g.abs().sum() > 0) for g in grads), f"{name} got no signal"

    assert logprobs.shape == values.shape == entropies.shape == (3,)


def test_chunks_are_scored_independently():
    """Two chunks: each starts from its own recorded memories."""
    net = StubNet()
    data = [make_data(trunk=torch.full((1, DIM), float(i))) for i in range(4)]
    actions = torch.tensor([0, 1, 0, 1])

    score_chunks(net, [[0, 1], [2, 3]], data, actions)

    assert net.seen[0]["trunk"] is data[0].memory["trunk"]
    assert net.seen[2]["trunk"] is data[2].memory["trunk"]  # second chunk restarts
    assert torch.equal(data[1].memory["trunk"], torch.full((1, DIM), 1.0))


def test_stored_memory_is_replayable_in_autograd():
    """The memory stored during rollout must be plain tensors.

    Rollout forwards run under ``inference_mode``; tensors *created* there stay
    "inference tensors" and autograd refuses to save them for backward. Storing
    therefore clones outside that context, and this pins it, because the buffer is
    replayed with autograd.
    """
    with torch.inference_mode():
        produced = {"trunk": torch.randn(1, DIM)}
    assert produced["trunk"].is_inference()

    stored = _stored(produced)
    assert not stored["trunk"].is_inference(), "would break truncated-BPTT replay"
    assert not stored["trunk"].requires_grad

    lin = nn.Linear(DIM, DIM)
    lin(stored["trunk"]).sum().backward()
    assert lin.weight.grad is not None


# --- trunk registry -------------------------------------------------------


class StubRoot(nn.Module):
    """Stand-in for ``RootNetwork``: the stub trunk ignores its output."""

    def __init__(
        self,
        name,
        feature_dim,
        condition_gat,
        hyperedge,
        statistics,
        pair_norm=False,
        rotation=True,
    ):
        super().__init__()
        self.name = name

    def forward(self, data):
        return None


class StubTrunk(nn.Module):
    """Stand-in for ``TruncNetwork``: only the contract ``Network`` relies on."""

    def __init__(
        self, name, feature_dim, option_transformer, summary_sage, pair_norm, memory
    ):
        super().__init__()
        self.name = name
        self.memory_enabled = memory
        self.lin = nn.Linear(feature_dim, feature_dim)
        self.calls = 0

    def forward(self, data, root_out):
        self.calls += 1
        dim = self.lin.in_features
        return TruncOutput(
            option=self.lin(torch.zeros(3, dim)),
            state=torch.zeros(1, dim),
            memory=torch.zeros(1, dim) if self.memory_enabled else None,
        )


def entity_data(memory: dict[str, torch.Tensor] | None = None) -> HecaData:
    """Data with the entity rows (current + goal), options and the memory."""
    data = HecaData()
    data["entity"].x = torch.randn(4, DIM)  # 2 entities: current, goal
    data["entity"].type_ids = torch.zeros(4, dtype=torch.long)
    data["entity"].role_ids = torch.tensor(
        [ENRole.START.value] * 2 + [ENRole.GOAL.value] * 2
    )
    data["option"].x = torch.randn(3, DIM)
    data["option"].gated = torch.zeros(3)
    data.memory = dict(memory or {})
    return data


def test_network_trunk_layout_is_name_keyed():
    """Both layouts expose the same keyed API, and nothing is registered twice."""
    import heca.heca_gnn.network as net_mod

    real_trunc, real_root = net_mod.TruncNetwork, net_mod.RootNetwork
    net_mod.TruncNetwork = StubTrunk  # type: ignore[assignment]
    net_mod.RootNetwork = StubRoot  # type: ignore[assignment]
    try:

        def make_net(separate: bool, memory: bool = True) -> nn.Module:
            return net_mod.Network(
                net_mod.Network.Config(
                    feature_dim=DIM,
                    seperate_root=separate,
                    seperate_trunc=separate,
                    use_memory=memory,
                )
            )

        shared = make_net(separate=False)
        assert shared.memory_keys == ("shared",)
        assert shared.key_for("actor", "trunc") == "shared"
        assert shared.trunc_for("actor") is shared.trunc_for("critic")

        out = shared(entity_data())
        assert shared.truncs["shared"].calls == 1  # one pass, not one per role
        assert set(out.memory) == {"shared"}
        assert out.logits.shape == (1, 3) and out.value.shape == (1,)

        separate = make_net(separate=True)
        assert separate.memory_keys == ("actor", "critic")
        assert separate.trunc_for("actor") is not separate.trunc_for("critic")
        assert set(separate(entity_data()).memory) == {"actor", "critic"}

        # every trunk/root is registered exactly once: no aliased parameters, and
        # the state_dict stays name keyed (truncs.actor.* / roots.shared.* ...)
        for model in (shared, separate):
            names = [n for n, _ in model.named_parameters()]
            assert len(names) == len(set(names))
            ids = [id(p) for p in model.parameters()]
            assert len(ids) == len(set(ids))
            assert all(
                k.startswith(("truncs.", "roots."))
                for k in model.state_dict()
                if "lin" in k
            )

        no_mem = make_net(separate=True, memory=False)
        assert no_mem(entity_data()).memory == {}
    finally:
        net_mod.TruncNetwork, net_mod.RootNetwork = real_trunc, real_root
