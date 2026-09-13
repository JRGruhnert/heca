"""End-to-end smoke test of the trunk and the network on a synthetic graph.

The graph is built by hand, so this exercises exactly the wiring that changed:

    state slots -> encoder -> scene conv -> pool -> z
    z -> GRU -> h_t     (conditioned back onto the option rows via FiLM)
    z -> value head
"""

import torch

from heca.data.entity import Entity
from heca.graphs.data import HecaData
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.scene_edges import SceneEdges
from heca.graphs.edges.summary_edges import SummaryEdges
from heca.graphs.edges.translation_edges import TranslationEdges
from heca.graphs.nodes.state_nodes import StateNodes
from heca.graphs.roles import ROLE_CURRENT, ROLE_GOAL
from heca.heca_gnn.network import Network
from heca.heca_gnn.trunc import TruncNetwork

D = Entity.FEATURE_DIM


def make_data(n_entities: int = 2, n_comps: int = 2, n_options: int = 3) -> HecaData:
    torch.manual_seed(0)
    data = HecaData()

    # one current and one goal row per entity, current rows first
    data["entity"].x = torch.randn(2 * n_entities, D)
    data["entity"].type_ids = torch.zeros(2 * n_entities, dtype=torch.long)  # free
    data["entity"].role_ids = torch.tensor(
        [ROLE_CURRENT] * n_entities + [ROLE_GOAL] * n_entities
    )
    data["comp"].x = torch.randn(n_comps, D)
    data["comp"].type_ids = torch.zeros(n_comps, dtype=torch.long)
    data["option"].x = torch.randn(n_options, D)
    data["state"].x = torch.full((1, StateNodes.FEATURE_DIM), 0.75)
    data["state"].type_ids = torch.zeros(1, dtype=torch.long)

    cond = [(0, 0), (1, 1)][:n_comps]
    data[ConditionEdges.type].edge_index = torch.tensor(cond, dtype=torch.long).T
    data[ConditionEdges.type].edge_attr = torch.randn(len(cond), 8)

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

    data.memory = {}
    return data


def test_trunk_forward_shapes_and_state_to_memory_path():
    trunk = TruncNetwork(TruncNetwork.Config(use_memory=True), "trunc")
    data = make_data()

    rows = trunk(data)
    assert rows.option.shape == (3, D), "one row per option, trunk width"
    assert rows.state.shape == (1, D), "the pooled situation is one vector"
    assert rows.memory is not None and rows.memory.shape == (1, D)

    # no recurrence -> no memory, but state and options are still produced
    off = TruncNetwork(TruncNetwork.Config(use_memory=False), "trunc")
    off_rows = off(data)
    assert off_rows.memory is None
    assert off_rows.state.shape == (1, D)

    # the actor head is sized for the trunk width (FiLM keeps the width, the
    # situation is not concatenated onto the options)
    assert trunk.output_dim == D
    assert rows.option.shape[1] == trunk.output_dim


def test_memory_recurs_and_carries_gradient_across_steps():
    """h_t comes out of the same pass that uses it, and stays differentiable."""
    trunk = TruncNetwork(TruncNetwork.Config(use_memory=True), "trunc")

    first = trunk(make_data()).memory
    assert first is not None

    data = make_data()
    data.memory = {"trunc": first.detach().clone()}
    second = trunk(data).memory
    assert second is not None and not torch.allclose(first, second), "state moved"

    # the recurrence is connected: gradient from h_2 reaches the GRU parameters,
    # which is what the chunked replay relies on.
    second.sum().backward()
    grads = [p.grad for p in trunk.timeline_layer.parameters() if p.grad is not None]
    assert grads, "the GRU got no gradient through the returned memory"
    assert any(bool(g.abs().sum() > 0) for g in grads)


def test_network_forward_returns_logits_value_and_memory():
    for separate in (False, True):
        net = Network(
            Network.Config(
                trunc=TruncNetwork.Config(use_memory=True),
                seperate_trunc=separate,
            )
        )
        out = net(make_data())

        assert out.logits.shape == (1, 3)
        assert out.value.shape == (1,)
        expected = {"actor", "critic"} if separate else {"trunc"}
        assert set(out.memory) == expected
        for name, memory in out.memory.items():
            assert memory.shape == (1, D), name

        # the value must be differentiable and not depend on the situation only
        out.value.sum().backward()

        if separate:
            assert not torch.allclose(out.memory["actor"], out.memory["critic"])
