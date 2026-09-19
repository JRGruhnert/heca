from unittest import mock

import numpy as np

from heca.graphs.graph import JITTER_SCOPES, Graph

SHARED = [0.5, 0.5, 0.5]
PER_ENTITY = ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0])


class Dummy:
    """The scope logic only needs the entity names."""


def make_graph(scope: str, jitter: float = 0.5) -> Graph:
    return Graph(
        entities={name: Dummy() for name in ("a", "b", "c")},
        use_rotation=False,
        position_jitter=jitter,
        jitter_scope=scope,
    )


def scripted(*vectors):
    calls: list[tuple[float, float]] = []

    def fake(low, high, size=None):
        calls.append((low, high))
        return np.asarray(vectors[len(calls) - 1], dtype=np.float64)

    return fake, calls


def offsets_with(scope: str, *vectors, jitter: float = 0.5):
    fake, calls = scripted(*vectors)
    with mock.patch("numpy.random.uniform", side_effect=fake):
        offsets = make_graph(scope, jitter=jitter).jitter_offsets()
    return offsets, calls


def test_entity_scope_draws_once_per_entity():
    offsets, calls = offsets_with("entity", *PER_ENTITY)
    assert len(calls) == 3
    for name, expected in zip("abc", PER_ENTITY):
        assert np.allclose(offsets[name], expected), name
    assert not np.allclose(offsets["a"], offsets["b"]), "entities are not independent"


def test_scene_scope_shifts_the_whole_scene_alike():
    offsets, calls = offsets_with("scene", SHARED)
    assert len(calls) == 1, "the scene draw must happen once per build"
    for name, value in offsets.items():
        assert np.allclose(value, SHARED), name


def test_both_scope_adds_a_shared_and_an_independent_part():
    offsets, calls = offsets_with("both", SHARED, *PER_ENTITY)
    assert len(calls) == 4, "one shared draw plus one per entity"
    for name, independent in zip("abc", PER_ENTITY):
        assert np.allclose(offsets[name], np.asarray(SHARED) + independent), name
        assert np.allclose(offsets[name] - independent, SHARED), name
    assert not np.allclose(offsets["a"], offsets["b"]), "entities are not independent"


def test_the_draws_respect_the_configured_limit():
    _, calls = offsets_with("both", SHARED, *PER_ENTITY, jitter=0.25)
    assert calls and all((low, high) == (-0.25, 0.25) for low, high in calls)


def test_none_scope_draws_nothing_even_with_a_limit():
    offsets, calls = offsets_with("none", jitter=0.5)
    assert calls == [], "none must not draw"
    assert all(np.allclose(value, 0.0) for value in offsets.values())


def test_zero_jitter_draws_nothing():
    for scope in JITTER_SCOPES:
        offsets, calls = offsets_with(scope, jitter=0.0)
        assert calls == [], scope
        assert all(np.allclose(value, 0.0) for value in offsets.values()), scope


def test_an_unknown_scope_is_rejected():
    try:
        make_graph("group")
    except ValueError as exc:
        assert "jitter_scope" in str(exc)
    else:
        raise AssertionError("an unknown jitter scope was accepted")
