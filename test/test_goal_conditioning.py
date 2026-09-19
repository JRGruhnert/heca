from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import DEFAULT_TERMS
from heca.heca_gnn.modules.boundary import NoUpdateBlock
from heca.heca_gnn.modules.hyperedge import TypedHyperedgeLayer
from heca.heca_gnn.network import Network

CHANNELS = ("none", "hyperedge", "residual")


def condition_edge_width(net: Network) -> int:
    conv = net.roots[Network.SHARED].layers["condition"].conv
    return conv.lin.weight.shape[1] if hasattr(conv, "lin") else conv.edge_dim


def test_each_channel_wires_exactly_one_path():
    for channel in CHANNELS:
        net = Network(Network.Config(use_rotation=False, goal_conditioning=channel))
        layer = net.roots[Network.SHARED].layers["hyperedge"]

        if channel == "hyperedge":
            assert isinstance(layer, TypedHyperedgeLayer)
        else:
            assert isinstance(layer, NoUpdateBlock), channel

        expected = ConditionEdges.edge_dim(
            False, DEFAULT_TERMS, goal_residual=channel == "residual"
        )
        assert (
            condition_edge_width(net) == expected
        ), f"{channel}: block expects {condition_edge_width(net)}, graph provides {expected}"


def test_the_two_paths_do_not_share_a_configuration():
    residual = Network(Network.Config(use_rotation=False, goal_conditioning="residual"))
    assert isinstance(
        residual.roots[Network.SHARED].layers["hyperedge"], NoUpdateBlock
    ), "the residual channel must not also condition rows through the hyperedge"


def test_an_unknown_channel_is_rejected():
    try:
        Network(Network.Config(goal_conditioning="hyperedge+residual"))
    except ValueError as exc:
        assert "goal_conditioning" in str(exc)
    else:
        raise AssertionError("an unknown goal-conditioning channel was accepted")
