import numpy as np
import pytest
from PIL import Image

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.data.prismatic import PrismaticEntity
from heca.data.reference import Reference
from heca.data.revolute import RevoluteEntity
from heca.scenes.scene import Scene, reference_from_note, reference_note

# scene0/drawer0: the demos record travel from -0.1666 to 0.0014. The env declares
# (-0.16, 0.0) instead, and that pair is not read anywhere any more.
DEMO_RANGE = (-0.1666, 0.0014)
DECLARED_MIN = -0.16

# The widest revolute travel in the shipped datasets (faucet0, scene10).
FAUCET_ANGLE = 1.80


def prismatic() -> PrismaticEntity:
    """A slide entity with no state of its own: the scene hands it its travel."""
    return PrismaticEntity.get(PrismaticEntity.Config())  # type: ignore[return-value]


def revolute() -> RevoluteEntity:
    return RevoluteEntity.get(RevoluteEntity.Config())  # type: ignore[return-value]


def drawer_obs(sca: float) -> dict:
    """The recorded keys ``extra_part`` reads, in the dataset's own units (m)."""
    return {"heca_drawer0_sca": np.array([sca])}


def segment_positions(
    fraction: float, limits: tuple[float, float] = DEMO_RANGE
) -> dict:
    """Reference keypoints for a slide along x, current at ``fraction`` of it."""
    start, end = 0.0, limits[1] - limits[0]
    return {
        "min": np.array([start, 0.0, 0.0]),
        "max": np.array([start + end, 0.0, 0.0]),
        "current": np.array([start + fraction * end, 0.0, 0.0]),
    }


def dc_scene(label: str, extra: np.ndarray, pos=None) -> DCScene:
    value = np.concatenate(
        [np.zeros(3) if pos is None else pos, np.zeros(3), extra, [0.0]]
    )
    return DCScene({label: DCEntity(value=value, feature=np.zeros(1))})


def arc_positions(angle: float, radius: float = 0.1, half_arc: float = 0.6) -> dict:
    """Three references on a circle in the xy-plane, ``mid`` at angle 0."""
    centre = np.zeros(3)

    def on(phi: float) -> np.ndarray:
        return centre + radius * np.array([np.cos(phi), np.sin(phi), 0.0])

    return {
        "min": on(-half_arc),
        "mid": on(0.0),
        "max": on(half_arc),
        "current": on(angle),
    }


# --- prismatic: a normalized slide fraction ---------------------------------


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75, 1.0])
def test_prismatic_paths_agree_inside_the_range(fraction: float):
    """Same slide state, same extra -- whether read from the joint or the image."""
    entity = prismatic()
    sca = DEMO_RANGE[0] + fraction * (DEMO_RANGE[1] - DEMO_RANGE[0])

    gt = entity.extra_part("drawer0", drawer_obs(sca), DEMO_RANGE).ravel()
    vision = entity.extra_from_references("drawer0", segment_positions(fraction))

    assert np.allclose(gt, vision, atol=1e-9)
    assert np.allclose(gt, [2.0 * fraction - 1.0])


def test_prismatic_normalizes_with_the_demos_not_with_the_declared_limits():
    """The env's declared minimum is not the -1 end of the travel."""
    entity = prismatic()

    assert np.isclose(
        entity.extra_part("drawer0", drawer_obs(DECLARED_MIN), DEMO_RANGE).ravel()[0],
        -0.9214,
        atol=1e-4,
    )


def test_the_travel_is_an_argument_not_entity_state():
    """The readout is a function of what it is handed, and nothing else.

    Entities are rebuilt from their config on every lookup, so anything stored on
    one is invisible to the next -- which is why the scene owns the travel and
    passes it. A single entity answering differently for two different travels is
    what that looks like when it works.
    """
    entity = prismatic()
    assert entity is not prismatic(), "each lookup builds its own view"

    declared = (-0.16, 0.0)
    for sca in (-0.16, -0.08):
        want = 2.0 * (sca - declared[0]) / (declared[1] - declared[0]) - 1.0
        got = entity.extra_part("drawer0", drawer_obs(sca), DEMO_RANGE).ravel()[0]
        assert entity.extra_part("drawer0", drawer_obs(sca), declared).ravel()[
            0
        ] == pytest.approx(want)
        assert not np.isclose(got, want), "the travel decided, not the entity"


def test_prismatic_extras_stay_in_the_normalized_range():
    """Both paths clip, so a slide past its recorded end cannot enter unclipped."""
    entity = prismatic()

    gt = entity.extra_part(
        "drawer0", drawer_obs(DEMO_RANGE[0] - 0.02), DEMO_RANGE
    ).ravel()
    assert np.isclose(gt[0], -1.0), "a slide past the minimum clips to the minimum"

    vision = np.array(
        [
            entity.extra_from_references("drawer0", segment_positions(f))
            for f in (-0.05, 0.5, 1.05)
        ]
    ).ravel()
    assert np.all(vision >= -1.0) and np.all(vision <= 1.0)


def test_a_slide_without_a_travel_is_refused():
    """Cropping the readout would silently use the wrong ends; say so instead."""
    with pytest.raises(ValueError):
        prismatic().extra_part("drawer0", drawer_obs(-0.08))


def test_prismatic_env_value_sends_the_point_and_no_joint_value():
    """The command side carries the tracked point, never a converted joint value.

    ogbench turns ``heca_<label>_pos`` into the joint coordinate itself, from the
    slide's own axis, so the command needs no travel of any kind: the demo pair is
    not read, and the fraction cannot be inverted back into metres.
    """
    entity = prismatic()

    for sca in (DEMO_RANGE[0], -0.08, DEMO_RANGE[1]):
        extra = entity.extra_part("drawer0", drawer_obs(sca), DEMO_RANGE).ravel()
        sent = entity.env_state_value(
            "drawer0", dc_scene("drawer0", extra, pos=np.array([1.0, 2.0, 3.0]))
        )

        assert set(sent) == {"heca_drawer0_pos", "heca_drawer0_rot", "heca_drawer0_ste"}
        assert "heca_drawer0_sca" not in sent
        assert np.allclose(
            sent["heca_drawer0_pos"], [1.0, 2.0, 3.0]
        ), "the point itself"

    # the vision path's case: no travel was ever derived, and the command side
    # does not care
    assert entity.env_state_value("drawer0", dc_scene("drawer0", np.zeros(1)))


def test_prismatic_extra_values_read_the_demos():
    """One expert's demos give the two ends of the travel, over all its frames."""
    demos = {"heca_drawer0_sca": np.array([[-0.05], [DEMO_RANGE[0]], [0.0014]])}

    assert prismatic().extra_values("drawer0", demos) == DEMO_RANGE


# --- the ends the encoder derives -------------------------------------------


class _PooledScene(Scene):
    """A scene stripped to the position pool, so the aggregation runs for real."""

    def __init__(self):
        super().__init__(Scene.Config(label="test", tag="test", folder="test"))

    @property
    def entities(self) -> dict[str, Entity]:
        return {"drawer0": prismatic()}

    def _load(self, path) -> bool:
        return True

    def _save(self, path) -> bool:
        return True

    def _step(self, action):  # pragma: no cover - the pool never steps
        raise NotImplementedError

    def _step_virt(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def load_dataset(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError

    def to_dc_scene(self, obs):  # pragma: no cover
        raise NotImplementedError

    def to_td_image(self, obs):  # pragma: no cover
        raise NotImplementedError

    def to_np_image(self, obs):  # pragma: no cover
        raise NotImplementedError


def test_the_pool_collects_points_one_frame_at_a_time():
    """A frame's keypoint is a ``(3,)`` point, and a pool of them has to stay 2-D.

    ``np.concatenate`` over a list of ``(3,)`` arrays flattens them into one
    vector, which the derivation then reads as a single broken point. This is the
    shape the encoder hands over, frame by frame.
    """
    scene = _PooledScene()
    for point in ([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]):
        scene.add_extra_positions("drawer0", np.array(point))

    pooled = np.concatenate(scene.extra_positions["drawer0"], axis=0)
    assert pooled.shape == (3, 3), "three frames, three points"

    scene.finish_extras()

    assert set(scene.extra_references["drawer0"]) == {"min", "max"}
    assert np.allclose(scene.extra_references["drawer0"]["min"], [0.0, 0.0, 0.0])
    assert np.allclose(scene.extra_references["drawer0"]["max"], [1.0, 0.0, 0.0])
    assert scene.extra_positions == {}, "the pool is consumed, not left behind"


def test_a_flat_pool_is_refused_rather_than_read_as_one_point():
    entity = prismatic()

    with pytest.raises(ValueError):
        entity.reference_positions(np.zeros(9))
    with pytest.raises(ValueError):
        entity.reference_positions(np.zeros((3, 2)))


def test_reference_positions_find_the_two_ends_of_a_travel():
    """A slide travels a line, so the ends are where the pooled positions cluster.

    The pooled frames are the ends of the demos, and the drawer is opened and
    closed several times, so both ends carry several frames.
    """
    entity = prismatic()
    opened = np.stack(
        [np.array([0.33, -0.07 - 0.15 * t, 0.08]) for t in (0.0, 0.02, 0.99)]
    )
    closed = np.stack([np.array([0.33, -0.07 - 0.01 * t, 0.08]) for t in (0.0, 0.03)])
    positions = np.concatenate([opened, closed], axis=0)

    found = entity.reference_positions(positions)

    assert set(found) == {"min", "max"}
    span = float(np.linalg.norm(found["max"] - found["min"]))
    assert np.isclose(span, 0.15, atol=0.005), "the distance the drawer travelled"
    for end in ("min", "max"):
        assert np.allclose(found[end][[0, 2]], [0.33, 0.08]), "ends stay on the slide"


def test_one_badly_tracked_frame_does_not_set_the_travel():
    """Ends from a median, not from an argmax.

    A frame the tracker lost lands far from the slide; the furthest-apart *pair*
    of pooled points would be that frame against the other end, which sets the
    span -- and with it the scale of every fraction the vision path reads. On the
    real scene0 demos that produced a drawer span 4.9x its actual travel.
    """
    entity = prismatic()
    ends = np.array([[0.33, -0.07, 0.08], [0.33, -0.22, 0.08]])
    # a demo's worth of frames at each end, as the derivation pools them, plus
    # one frame the tracker lost
    pool = np.concatenate(
        [
            np.tile(ends[0], (20, 1)),
            np.tile(ends[1], (20, 1)),
            np.array([[0.33, 0.6, 0.08]]),
        ]
    )

    found = entity.reference_positions(pool)
    span = float(np.linalg.norm(found["max"] - found["min"]))

    assert np.isclose(span, 0.15, atol=1e-9), "the stray frame is ignored"
    farthest = np.linalg.norm(pool[:, None] - pool[None, :], axis=-1).max()
    assert farthest > 0.7, "the pair the argmax would have taken is much wider"


def test_the_ends_are_stable_whatever_order_the_frames_arrive_in():
    """Which end is ``min`` is a convention, and it has to hold every time.

    The joint's direction is not read, so the order is fixed by the geometry
    alone. An eigenvector's sign is arbitrary, so without pinning it the same
    travel could come out either way round from one pool to the next -- and that
    flips the sign of every fraction the vision path reads.
    """
    entity = prismatic()
    near = np.array([0.0, 0.0, 0.0])
    far = np.array([0.4, 0.0, 0.0])

    forward = entity.reference_positions(np.stack([near, far, near, far]))
    shifted = entity.reference_positions(
        np.stack([near + 7.0, far + 7.0, far + 7.0, near + 7.0])
    )

    assert np.allclose(forward["min"], near) and np.allclose(forward["max"], far)
    assert np.allclose(shifted["min"], near + 7.0), "same travel, same order"
    assert np.allclose(shifted["max"], far + 7.0)
    assert np.isclose(np.linalg.norm(forward["max"] - forward["min"]), 0.4)


def test_a_travel_with_no_extent_is_refused():
    """One point repeated is no slide at all, and a fraction needs two ends."""
    entity = prismatic()

    with pytest.raises(ValueError):
        entity.reference_positions(np.tile(np.array([[0.3, 0.1, 0.2]]), (4, 1)))
    with pytest.raises(ValueError):
        entity.reference_positions(np.zeros((1, 3)))


def test_only_the_prismatic_entity_derives_its_references():
    """A slide takes no clicks; a hinge keeps its three and derives nothing."""
    assert prismatic().reference_names == ()
    assert revolute().reference_names == ("min", "mid", "max")
    assert revolute().reference_positions(np.zeros((4, 3))) is None


# --- revolute: a point on the unit circle -----------------------------------


@pytest.mark.parametrize("angle", [-np.pi, -FAUCET_ANGLE, -0.3, 0.0, 0.7, FAUCET_ANGLE])
def test_revolute_extras_lie_on_the_unit_circle(angle: float):
    entity = revolute()
    gt = entity.extra_part("faucet0", {"heca_faucet0_ang": np.array([angle])}).ravel()
    vision = entity.extra_from_references("faucet0", arc_positions(angle))

    assert np.allclose(gt, [np.sin(angle), np.cos(angle)])
    assert np.isclose(np.linalg.norm(gt), 1.0)
    assert np.isclose(np.linalg.norm(vision), 1.0)


@pytest.mark.parametrize("angle", [-0.5, 0.0, 0.3, 1.0])
def test_revolute_mid_reference_is_the_zero(angle: float):
    """The hand-clicked ``mid`` reference sets angle 0, not the env's convention.

    ``e1`` points from the fitted circle centre to ``mid`` and
    ``e2 = normal x e1``, so the three references fix both the zero and the sign
    of the vision angle. Nothing reconciles that gauge with the recorded angle.
    """
    entity = revolute()
    got = entity.extra_from_references("faucet0", arc_positions(angle))
    assert np.allclose(got, [np.sin(angle), np.cos(angle)], atol=1e-9)


def test_revolute_swapping_the_arc_references_flips_the_sign():
    """Which end is ``min`` and which is ``max`` decides the sign of the angle."""
    entity = revolute()
    forward = arc_positions(0.4)
    swapped = {**forward, "min": forward["max"], "max": forward["min"]}

    got = entity.extra_from_references("faucet0", swapped)
    assert np.allclose(got, [np.sin(-0.4), np.cos(-0.4)], atol=1e-9)


def test_revolute_env_value_is_the_inverse_of_extra_part():
    """Unlike the slide, the angle needs no limits, so the round trip is exact."""
    entity = revolute()

    for angle in (-FAUCET_ANGLE, -0.2, 0.0, 0.9, FAUCET_ANGLE):
        extra = entity.extra_part(
            "faucet0", {"heca_faucet0_ang": np.array([angle])}
        ).ravel()
        scene = dc_scene("faucet0", extra)

        got = entity.env_state_value("faucet0", scene)["heca_faucet0_ang"]
        assert np.isclose(got, angle, atol=1e-9)


def test_revolute_sanitize_puts_a_sampled_extra_back_on_the_circle():
    """A Gaussian sample lands off the circle; ``sanitize_value`` rescales it."""
    entity = revolute()
    off_circle = np.array([0.8, 0.9])  # |.| = 1.20

    value = np.zeros(entity.pose_dim + 1)
    value[Entity.POS_DIM + Entity.ROT_DIM : -1] = off_circle
    out = entity.sanitize_value(value)

    assert np.allclose(out[:6], value[:6]), "only the extra block is touched"
    assert np.isclose(np.linalg.norm(out[6:8]), 1.0)
    assert np.allclose(out[6:8], off_circle / np.linalg.norm(off_circle))


def test_revolute_gnn_features_keep_off_circle_extras():
    """``comp_feature`` writes the fitted values as they are.

    Sampling is repaired by ``sanitize_value``, the features are not, so the
    graph network sees sin/cos pairs that a caller would never execute.
    """
    entity = revolute()
    extra = np.array([0.8, 0.9])
    up = {
        "weights": np.array([1.0]),
        "measurement": {
            "pose": {
                "means": np.concatenate([np.zeros(6), extra])[None, :],
                "covariances": np.full((1, entity.pose_dim), 1e-4),
            },
            "state": {"pis": np.array([[1.0]])},
        },
    }

    feat, _ = entity.comp_feature(up)
    got = feat[0, Entity.LAYOUT["extra"].mean(2)]

    assert np.allclose(got, extra), "documented gap: no renormalization here"
    assert not np.isclose(np.linalg.norm(got), 1.0)


def test_revolute_direction_key_needs_travel_inside_the_seam():
    """``make_agent_key`` compares raw angles, so it needs travel in (-pi, pi].

    The widest shipped faucet travel is +-1.80 rad, comfortably inside the seam,
    so the a_b/b_a label is the true direction. Past +-pi the same forward motion
    reads as backwards: +3.0 rad is -3.28 rad wrapped, so -3.0 is *ahead* of it,
    yet the comparison sees the plain numbers and answers "b_a".
    """
    entity = revolute()

    def key(start: float, end: float) -> str:
        obs = {"heca_faucet0_ang": np.array([[start], [end]])}
        return entity.make_agent_key("faucet0", obs, 0, 1)

    assert key(-FAUCET_ANGLE, FAUCET_ANGLE) == "faucet0_a_b"
    assert key(FAUCET_ANGLE, -FAUCET_ANGLE) == "faucet0_b_a"
    assert key(3.0, -3.0) == "faucet0_b_a", "documented gap: crosses the seam"


# --- what the annotation stores beside a reference --------------------------


def reference(xyz=(0.0, 0.0, 0.0)) -> Reference:
    return Reference(
        image=Image.new("RGB", (2, 2)),
        x=1,
        y=1,
        xyz=np.asarray(xyz, dtype=np.float64),
    )


def test_the_annotation_holds_only_the_point_and_its_pixel():
    note = reference_note(reference(xyz=(1.0, 2.0, 3.0)))
    assert note == {"x": 1, "y": 1, "xyz": [1.0, 2.0, 3.0]}, "no joint value stored"

    back = reference_from_note(note, Image.new("RGB", (2, 2)))
    assert np.allclose(back.xyz, [1.0, 2.0, 3.0])
    assert back.x == 1 and back.y == 1
