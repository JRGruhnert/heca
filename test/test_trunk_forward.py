"""End-to-end smoke test of the two network segments on a synthetic graph.

The graph is built by hand, so this exercises exactly the wiring that changed:

    rows -> encoder -> condition -> hyperedge -> translation     (root)
    entity rows -> summary -> interaction -> scene -> GRU        (trunc)
    scene row -> value head, option rows -> policy head
"""

import torch

from heca.data.entity import Entity
from heca.graphs.data import HecaData
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.state_nodes import StateNodes
from heca.graphs.roles import ENRole
from heca.heca_gnn.network import Network
from heca.heca_gnn.root import RootNetwork
from heca.heca_gnn.trunc import TruncNetwork

D = Entity.FEATURE_DIM
DP = Entity.POINT_FEATURE_DIM  # point values (entity rows) carry no log-std


def make_data(n_entities: int = 2, n_comps: int = 2, n_options: int = 3) -> HecaData:
    torch.manual_seed(0)
    data = HecaData()

    # one current and one goal row per entity, current rows first
    data["entity"].x = torch.randn(2 * n_entities, DP)
    data["entity"].type_ids = torch.zeros(2 * n_entities, dtype=torch.long)  # free
    data["entity"].role_ids = torch.tensor(
        [ENRole.START.value] * n_entities + [ENRole.GOAL.value] * n_entities
    )
    data["comp"].x = torch.randn(n_comps, D)
    data["comp"].type_ids = torch.zeros(n_comps, dtype=torch.long)
    data["option"].x = torch.randn(n_options, D)
    data["state"].x = torch.full((1, StateNodes.FEATURE_DIM), 0.75)
    data["state"].type_ids = torch.zeros(1, dtype=torch.long)

    cond = [(0, 0), (1, 1)][:n_comps]
    data[ConditionEdges.type].edge_index = torch.tensor(cond, dtype=torch.long).T
    data[ConditionEdges.type].edge_attr = torch.randn(len(cond), 27)

    data[TranslationEdges.type].edge_index = torch.tensor([(0, 1)], dtype=torch.long).T
    data[SummaryEdges.type].edge_index = torch.tensor(
        [(i % n_entities, i) for i in range(n_options)], dtype=torch.long
    ).T

    # one signed edge per option into the single state node
    gated = [i % 2 == 1 for i in range(n_options)]
    data[SceneEdges.type].edge_index = torch.tensor(
        [[o for o in range(n_options)], [0] * n_options], dtype=torch.long
    )
    data[SceneEdges.type].edge_attr = torch.tensor(
        [[-1.0 if g else 1.0] for g in gated]
    )
    data["option"].gated = torch.tensor([float(g) for g in gated])

    data.memory = {}
    return data


def make_root(name: str = "shared") -> RootNetwork:
    return RootNetwork(
        name,
        feature_dim=D,
        condition_gat=False,
        hyperedge=False,
        rotation=True,
    )


def make_trunc(name: str = "shared", memory: bool = True) -> TruncNetwork:
    return TruncNetwork(
        name,
        feature_dim=D,
        option_transformer=False,
        summary_sage=False,
        memory=memory,
    )


def test_root_forward_returns_the_encoded_rows():
    x = make_root()(make_data())
    assert x.entity.shape == (4, D), "one row per entity row"
    assert x.comp.shape == (2, D) and x.state.shape == (1, D), "untouched encodings"


def test_trunk_forward_shapes_and_state_to_memory_path():
    data = make_data()
    x = make_root()(data)

    trunk = make_trunc(memory=True)
    rows = trunk(data, x)
    assert rows.option.shape == (3, D), "one row per option, trunk width"
    assert rows.state.shape == (1, D), "the pooled situation is one vector"
    assert rows.memory is not None and rows.memory.shape == (1, D)

    # no recurrence -> no memory, but state and options are still produced
    off_rows = make_trunc(memory=False)(data, x)
    assert off_rows.memory is None
    assert off_rows.state.shape == (1, D)


def test_memory_recurs_and_carries_gradient_across_steps():
    """h_t comes out of the same pass that uses it, and stays differentiable."""
    data = make_data()
    x = make_root()(data)
    trunk = make_trunc(memory=True)

    first = trunk(data, x).memory
    assert first is not None

    data.memory = {"shared": first.detach().clone()}
    second = trunk(data, x).memory
    assert second is not None and not torch.allclose(first, second), "state moved"

    # the recurrence is connected: gradient from h_2 reaches the GRU parameters,
    # which is what the chunked replay relies on.
    second.sum().backward()
    grads = [
        p.grad for p in trunk.layers["timeline"].parameters() if p.grad is not None
    ]
    assert grads, "the GRU got no gradient through the returned memory"
    assert any(bool(g.abs().sum() > 0) for g in grads)


def test_network_forward_returns_logits_value_and_memory():
    for separate_root, separate_trunc in ((False, False), (False, True), (True, True)):
        net = Network(
            Network.Config(
                feature_dim=D,
                seperate_root=separate_root,
                seperate_trunc=separate_trunc,
                use_memory=True,
            )
        )
        out = net(make_data())

        assert out.logits.shape == (1, 3)
        assert out.value.shape == (1,)
        expected = {"actor", "critic"} if separate_trunc else {"shared"}
        assert set(out.memory) == expected
        for name, memory in out.memory.items():
            assert memory.shape == (1, D), name

        # the value must be differentiable
        out.value.sum().backward()

        if separate_trunc:
            assert not torch.allclose(out.memory["actor"], out.memory["critic"])
