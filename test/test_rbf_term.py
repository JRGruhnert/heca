"""The ``rbf`` edge term: Gaussian bumps over the acceptance-normalised residual.

``rho = r / r_max`` is the residual distance in units of the chi bound the gate
itself tests against, so the grid is tuned so that a centre sits exactly on
``rho = 1`` - the "fits / does not fit" boundary.  The term is a fixed feature
map: nothing about it is learned, only the weights that mix it.

The checks are written against the class attributes (``EdgeSet.rho_max``,
``EdgeSet.rbf_centers``) rather than hard-coded numbers, so retuning the grid
keeps them meaningful - except the boundary invariant, which is the point of the
design and must fail loudly if a retune breaks it.
"""

import numpy as np

from heca.data.entity import Entity, _chi_sqrt
from heca.data.static import StaticEntity
from heca.graphs.edges.condition_edges import ConditionEdges
from heca.graphs.edges.edge_set import EdgeSet

RHO_MAX = EdgeSet.rho_max
K = EdgeSet.rbf_centers
CENTRES = np.linspace(0.0, RHO_MAX, K)
SPACING = RHO_MAX / (K - 1)


def residue() -> np.ndarray:
    """A single all-zero src/dst pair, so every residual is zero."""
    src = np.zeros((1, Entity.FEATURE_DIM), dtype=np.float32)
    dst = np.zeros((1, Entity.POINT_FEATURE_DIM), dtype=np.float32)
    return src, dst


def test_a_centre_sits_on_the_boundary() -> None:
    """rho = 1 must be a centre: that is what makes the boundary readable."""
    assert np.any(np.isclose(CENTRES, 1.0)), (
        f"no centre on rho = 1 with rho_max={RHO_MAX}, K={K}; use "
        "K = 4 * rho_max + 1 to keep the 0.25 spacing and the boundary centre"
    )


def test_rbf_width_matches_the_graph_and_the_network() -> None:
    assert K == CENTRES.size
    assert EdgeSet._term_width("rbf", True) == EdgeSet._term_width("rbf", False) == K
    assert EdgeSet.residual_dim(("rbf",), False) == K
    assert ConditionEdges.edge_dim(False, ("rbf",)) == K + 1
    assert ConditionEdges.edge_dim(True, ("rbf",)) == K + 1


def test_no_gap_between_neighbours() -> None:
    """sigma = spacing, so no value on [0, rho_max] falls between two centres."""
    edges = ConditionEdges()
    worst = 1.0
    for rho in np.linspace(0.0, RHO_MAX, 201):
        worst = min(worst, float(edges.rbf_features(np.array([rho])).max()))
    assert worst > 0.05, f"a value on [0, {RHO_MAX}] is invisible (best bump {worst:.3f})"
    # sanity: the midpoint response of the analytic formula
    assert np.isclose(np.exp(-((SPACING / 2) ** 2) / (2 * SPACING**2)), 0.8825, atol=1e-3)


def test_boundary_peak_and_far_outside() -> None:
    edges = ConditionEdges()
    index = int(np.argmin(np.abs(CENTRES - 1.0)))
    code = edges.rbf_features(np.array([0.0, 1.0, 2.0 * RHO_MAX]))
    assert code[0, 0] == 1.0
    assert code[1, index] == 1.0
    assert np.allclose(code[2], 0.0, atol=1e-4), "far outside must collapse to zeros"
    assert np.all(code >= 0.0) and np.all(code <= 1.0)
    # the strongest bump tracks the value, and outside the boundary the last
    # centres keep responding instead of the code going straight to zero
    order = [int(np.argmax(edges.rbf_features(np.array([rho]))[0])) for rho in CENTRES]
    assert order == sorted(order)
    assert float(edges.rbf_features(np.array([1.25])).max()) > 0.9


def test_rbf_requires_the_acceptance_radius() -> None:
    edges = ConditionEdges()
    src, dst = residue()
    try:
        edges.residual_batch(src, dst, terms=("rbf",))
    except ValueError as exc:
        assert "acceptance radius" in str(exc)
    else:
        raise AssertionError("rbf without a radius must raise")

    near = edges.residual_batch(src, dst, terms=("rbf",), radius=np.full(1, 4.0))
    assert near.shape == (1, K)
    assert np.isclose(near[0, 0], 1.0), "zero residual sits on the first centre"


def test_acceptance_radius_is_the_gate_bound() -> None:
    entity = StaticEntity.get(StaticEntity.Config(type_id=1, n_states=2))  # type: ignore[return-value]
    for rotation in (True, False):
        expected = _chi_sqrt(
            entity.cfg.z_quantile_joint, len(entity.pose_groups(rotation))
        )
        assert np.isclose(entity.acceptance_radius(rotation), expected)
    # dropping rotation drops those groups, so the bound moves to the remaining dof
    assert entity.acceptance_radius(False) < entity.acceptance_radius(True)


def test_rbf_block_in_the_loop() -> None:
    """The term has to survive the real edge builder, shape included."""
    edges = ConditionEdges()
    src = np.zeros((3, Entity.FEATURE_DIM), dtype=np.float32)
    dst = np.zeros((3, Entity.POINT_FEATURE_DIM), dtype=np.float32)
    out = edges.residual_batch(src, dst, terms=("rbf",), radius=np.ones(3))
    assert out.shape == (3, K)
    assert np.allclose(out[:, 0], 1.0)
