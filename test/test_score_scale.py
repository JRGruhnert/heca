"""The two score-softening knobs: variance scale and state smoothing.

Both only affect the *tests* (``score_single``/``prepare_single``/``score_state``),
so they need no refit and also apply to conditions loaded from a fit cache. The
data here is synthetic, so the tests run without the scene datasets.
"""

import numpy as np
import torch

from heca.data.entity import _chi_sqrt
from heca.data.static import StaticEntity

DIM = 3  # StaticEntity: pos (3) + rotation (3) = 6 pose dims, state is last


def make_entity(scale: float = 1.0, smoothing: float = 0.0) -> StaticEntity:
    cfg = StaticEntity.Config(
        type_id=1,
        n_states=2,
        score_variance_scale=scale,
        score_state_smoothing=smoothing,
    )
    return StaticEntity.get(cfg)  # type: ignore[return-value]


def make_params(entity: StaticEntity, state_1_prob: float = 0.0) -> dict:
    pose_dim = entity.pose_dim
    return {
        "weights": np.array([1.0]),
        "measurement": {
            "pose": {
                "means": np.zeros((1, pose_dim)),
                "covariances": np.full((1, pose_dim), 1e-4),
            },
            "state": {"pis": np.array([[1.0 - state_1_prob, state_1_prob]])},
        },
    }


def make_value(entity: StaticEntity, offset: float, state: int) -> np.ndarray:
    value = np.zeros(entity.pose_dim + 1)
    value[: entity.pose_dim] = offset
    value[-1] = state
    return value


def test_defaults_leave_the_scoring_parameters_untouched():
    entity = make_entity()
    up = make_params(entity)
    scaled = entity.scoring_parameters(up)
    plain = entity.secure_mix_parameters(up, add_variance=True)

    assert np.array_equal(
        scaled["measurement"]["pose"]["covariances"],
        plain["measurement"]["pose"]["covariances"],
    )
    assert np.array_equal(
        scaled["measurement"]["state"]["pis"], plain["measurement"]["state"]["pis"]
    )
    assert entity.score_scale == 1.0 and entity.score_smoothing == 0.0


def test_variance_scale_widens_the_acceptance_ellipse():
    """A value far outside a tight fit passes once the covariance is inflated."""
    entity = make_entity()
    up = make_params(entity)
    far = make_value(entity, offset=0.05, state=0)  # 5 sigma per dimension

    chi = _chi_sqrt(entity.cfg.z_quantile_joint, entity.pose_dof)
    _, z, _ = entity._best_component(
        entity.model_value(far)[:-1], entity.scoring_parameters(up)
    )
    assert z > chi, "the test value has to start outside the fitted region"
    assert not entity.score_single(far, up, True)

    entity.cfg.score_variance_scale = 100.0  # 10x sigma, z shrinks by 10
    assert entity.score_scale == 100.0
    assert entity.score_single(far, up, True)


def test_state_smoothing_lifts_an_unobserved_state():
    """A state with fitted probability 0 is rejected, and accepted after mixing."""
    entity = make_entity()
    up = make_params(entity, state_1_prob=0.0)
    unseen = make_value(entity, offset=0.0, state=1)
    assert not entity.score_single(unseen, up, True), "p is exactly 0 for state 1"

    entity.cfg.score_state_smoothing = 0.5  # half uniform prior
    pis = entity.scoring_parameters(up)["measurement"]["state"]["pis"]
    assert np.isclose(pis[0][1], 0.25), "mixing has to give the state a mass"
    assert entity.score_single(unseen, up, True)


def test_state_floor_can_disable_the_state_test():
    entity = make_entity()
    up = make_params(entity, state_1_prob=0.0)
    unseen = make_value(entity, offset=0.0, state=1)
    assert not entity.score_single(unseen, up, True)

    entity.cfg.score_state_floor = 0.0
    assert entity.score_single(unseen, up, True)


def test_position_jitter_moves_only_the_position_block():
    """The augmentation helper shifts xyz per row and touches nothing else."""
    from heca.data.entity import Entity

    pos = Entity.POINT_LAYOUT["pos"].mean()
    offsets = np.array([[0.01, -0.02, 0.03], [0.01, -0.02, 0.03], [0.0, 0.0, 0.0]])

    x = np.zeros((3, Entity.POINT_FEATURE_DIM), dtype=np.float32)
    x[:, pos] = 1.0
    out = Entity.jitter_positions(x, offsets, Entity.POINT_LAYOUT)

    assert np.allclose(out[0, pos], [1.01, 0.98, 1.03])
    assert np.allclose(out[1, pos], out[0, pos]), "same entity, same offset"
    assert np.allclose(out[2, pos], [1.0, 1.0, 1.0]), "zero offset keeps the row"
    assert np.array_equal(out[:, : pos.start], x[:, : pos.start])
    assert np.array_equal(out[:, pos.stop :], x[:, pos.stop :])
    assert np.array_equal(x[:, pos], np.ones((3, 3))), "input must not be mutated"

    torch_x = torch.zeros(2, Entity.FEATURE_DIM)
    torch_out = Entity.jitter_positions(
        torch_x, np.zeros((2, 3), dtype=np.float32), Entity.LAYOUT
    )
    assert torch.equal(torch_out, torch_x) and torch_out is not torch_x
