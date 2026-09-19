import numpy as np
import torch

from heca.data.entity import Entity
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import DEFAULT_TERMS, EdgeSet
from heca.graphs.nodes.comp_nodes import CompNodes
from heca.graphs.nodes.entity_nodes import EntityNodes
from heca.graphs.nodes.node import CompNode, EntityNode
from heca.graphs.roles import ENMode, ENRole
from heca.heca_gnn.root import RootNetwork
from test.test_trunk_forward import D, make_data

ROT = False
COMP_DIM = Entity.FEATURE_DIM if ROT else Entity.NO_ROT_FEATURE_DIM
ROW_DIM = Entity.POINT_FEATURE_DIM if ROT else Entity.NO_ROT_POINT_FEATURE_DIM
LC = Entity.layout(ROT, logstd=True)
LP = Entity.layout(ROT, logstd=False)
POS = LP["pos"].mean()


class StubEntity:
    """Only the acceptance radius matters to the edge builder."""

    def acceptance_radius(self, rotation: bool) -> float:
        return 4.0


class StubCondition:
    def __init__(self, name: str):
        self.entities = {name: StubEntity()}


def comp_row(pos: np.ndarray | None = None) -> np.ndarray:
    """A component with unit spread, so z is the plain pose offset."""
    x = np.zeros(COMP_DIM, dtype=np.float32)
    if pos is not None:
        x[LC["pos"].mean()] = pos
    return x


def value_row(pos: np.ndarray) -> np.ndarray:
    x = np.zeros(ROW_DIM, dtype=np.float32)
    x[LP["pos"].mean()] = pos
    return x


def build_nodes(
    row_pos: np.ndarray, goal_pos: np.ndarray, with_goal: bool = True
) -> tuple[CompNodes, EntityNodes]:
    snset, dnset = CompNodes(), EntityNodes()
    snset.add("c0", CompNode(entity="e0", type_id=0, data=None, n_states=2, weight=0.5))
    snset.x = torch.from_numpy(np.stack([comp_row()]))
    dnset.add(
        "post",
        EntityNode(
            entity="e0",
            type_id=0,
            data=None,
            n_states=2,
            mode=ENMode.START,
            role=ENRole.POST,
            con=StubCondition("e0"),
        ),
    )
    dnset.items[0].sources["comp"] = {"c0"}
    rows = [value_row(row_pos)]
    if with_goal:
        dnset.add(
            "goal",
            EntityNode(
                entity="e0",
                type_id=0,
                data=None,
                n_states=2,
                mode=ENMode.GOAL,
                role=ENRole.GOAL,
            ),
        )
        rows.append(value_row(goal_pos))
    dnset.x = torch.from_numpy(np.stack(rows))
    return snset, dnset


def test_edge_dim_grows_by_exactly_one_residual_block():
    for terms in (("z",), ("rbf",), ("z", "rbf", "state")):
        for rotation in (True, False):
            one = ConditionEdges.edge_dim(rotation, terms, False)
            two = ConditionEdges.edge_dim(rotation, terms, True)
            assert two - 1 == 2 * (one - 1), (terms, rotation)


def test_the_second_block_tests_the_goal_not_the_rows_own_value():
    row = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    edges = ConditionEdges()
    edges.build(*build_nodes(row, goal), ROT, ("z",), goal_residual=True)

    width = EdgeSet.residual_dim(("z",), ROT)
    assert edges.edge_attr.shape == (1, ConditionEdges.edge_dim(ROT, ("z",), True))
    own, goal_block = edges.edge_attr[0, :width], edges.edge_attr[0, width : 2 * width]

    assert np.allclose(own[:3].numpy(), row), "first block is the row's own value"
    assert np.allclose(goal_block[:3].numpy(), goal), "second block is the goal"
    assert not np.allclose(own, goal_block), "the two blocks are not duplicates"


def test_the_goal_block_is_normalised_by_the_acceptance_radius():
    """The goal sits on the component mean, the row's own value far outside it."""
    goal = np.zeros(3, dtype=np.float32)
    row = np.array([0.0, 0.0, 50.0], dtype=np.float32)  # far beyond the radius
    edges = ConditionEdges()
    edges.build(*build_nodes(row, goal), ROT, ("rbf",), goal_residual=True)

    block = EdgeSet._term_width("rbf", ROT)
    own, goal_block = edges.edge_attr[0, :block], edges.edge_attr[0, block : 2 * block]
    assert np.isclose(goal_block[0], 1.0, atol=1e-6), "the goal is not accepted here"
    assert float(own.max()) < 0.9, "the row's own value should read as far outside"


def test_a_postcondition_row_that_already_holds_the_goal_reads_the_same_twice():
    """The goal-mode rows of a real graph are exactly this case."""
    goal = np.array([0.5, 0.5, 0.0], dtype=np.float32)
    edges = ConditionEdges()
    edges.build(*build_nodes(goal, goal), ROT, ("z",), goal_residual=True)

    width = EdgeSet.residual_dim(("z",), ROT)
    assert np.allclose(
        edges.edge_attr[0, :width].numpy(),
        edges.edge_attr[0, width : 2 * width].numpy(),
    )


def test_the_goal_half_is_live_on_precondition_rows():
    """A precondition row is where the goal-versus-current difference lives."""
    row = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    snset, dnset = build_nodes(row, goal)
    dnset.items[0].role = ENRole.PRE
    edges = ConditionEdges()
    edges.build(snset, dnset, ROT, ("z",), goal_residual=True)

    width = EdgeSet.residual_dim(("z",), ROT)
    own, goal_block = edges.edge_attr[0, :width], edges.edge_attr[0, width : 2 * width]
    assert not np.allclose(
        goal_block.numpy(), 0.0
    ), "a precondition row lost its goal half"
    assert np.allclose(goal_block[:3].numpy(), goal)


def test_the_two_halves_differ_by_the_goal_current_distance():
    """own - goal half == (current - goal) in the component's units, i.e. option C."""
    current = np.array([2.0, 1.0, -1.0], dtype=np.float32)
    goal = np.array([0.25, -0.5, 0.0], dtype=np.float32)
    snset, dnset = build_nodes(current, goal)  # the row's own value is the current one
    edges = ConditionEdges()
    edges.build(snset, dnset, ROT, ("z",), goal_residual=True)

    width = EdgeSet.residual_dim(("z",), ROT)
    own, goal_block = edges.edge_attr[0, :width], edges.edge_attr[0, width : 2 * width]
    assert np.allclose((own - goal_block)[:3].numpy(), current - goal)


def test_a_missing_goal_row_is_reported():
    edges = ConditionEdges()
    snset, dnset = build_nodes(
        np.zeros(3, dtype=np.float32), np.zeros(3, dtype=np.float32), with_goal=False
    )
    try:
        edges.build(snset, dnset, ROT, ("z",), goal_residual=True)
    except ValueError as exc:
        assert "goal row" in str(exc)
    else:
        raise AssertionError("a missing goal row must raise")


def test_the_block_consumes_the_wider_attribute():
    data = make_data()
    data[ConditionEdges.type].edge_attr = torch.randn(
        data[ConditionEdges.type].edge_index.shape[1],
        ConditionEdges.edge_dim(True, DEFAULT_TERMS, True),
    )
    root = RootNetwork(
        "shared",
        feature_dim=D,
        condition_gat=False,
        hyperedge=False,
        statistics=True,
        pair_norm=False,
        rotation=True,
        goal_residual=True,
    )
    x = root(data)
    assert x.entity.shape == (4, D)
    assert torch.isfinite(x.entity).all()
