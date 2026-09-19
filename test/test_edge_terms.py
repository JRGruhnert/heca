"""The condition-edge term selection: declared widths match the built feature.

Any subset of ``RESIDUAL_TERMS`` can be selected through ``Network.Config.edge_terms``,
and the graph and the network must agree on the resulting width - a mismatch is a
shape error deep inside the first conv.  These checks are synthetic: no scene needed.
"""

import numpy as np

from heca.data.entity import Entity
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import DEFAULT_TERMS, RESIDUAL_TERMS, EdgeSet

SUBSETS = [
    ("z",),
    ("rbf",),
    ("z", "state"),
    ("logz", "logp", "state"),
    ("z", "z2", "logz", "logp", "state"),
    RESIDUAL_TERMS,
    DEFAULT_TERMS,
]


def features(rotation: bool) -> tuple[np.ndarray, np.ndarray]:
    """Two synthetic rows, wide enough for either rotation mode."""
    comp = np.zeros((2, Entity.FEATURE_DIM), dtype=np.float32)
    entity = np.zeros((2, Entity.POINT_FEATURE_DIM), dtype=np.float32)
    return comp, entity


def test_declared_width_matches_the_built_feature() -> None:
    edges = ConditionEdges()
    for rotation in (True, False):
        comp, entity = features(rotation)
        for terms in SUBSETS:
            built = edges.residual_batch(
                comp,
                entity,
                use_rotation=rotation,
                terms=terms,
                radius=np.ones(2),
            )
            declared = EdgeSet.residual_dim(terms, rotation)
            assert built.shape == (2, declared), (
                f"terms={terms} rotation={rotation}: built {built.shape}, declared {declared}"
            )
            assert ConditionEdges.edge_dim(rotation, terms) == declared + 1


def test_term_widths_add_up() -> None:
    for rotation in (True, False):
        assert EdgeSet.residual_dim(RESIDUAL_TERMS, rotation) == sum(
            EdgeSet._term_width(term, rotation) for term in RESIDUAL_TERMS
        )
        # every term on its own is meaningful, and rbf is rotation independent
        for term in RESIDUAL_TERMS:
            assert EdgeSet._term_width(term, rotation) > 0
    assert EdgeSet._term_width("rbf", True) == EdgeSet._term_width("rbf", False)


def test_bad_selections_are_rejected() -> None:
    for bad in (("zz",), ("z", "nope"), ()):
        try:
            EdgeSet.residual_dim(bad, False)
        except ValueError:
            continue
        raise AssertionError(f"{bad} should be rejected")
