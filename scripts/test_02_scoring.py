"""Regression tests for the pre/post condition scoring functions.

Covers the fixes applied to ``heca.data.entity`` and ``heca.conditions.condition``:

  B1  ``score_single`` returns a normalized ``[0, 1]`` score instead of an
      unbounded probability density (static entities used to score ~1e15 at
      their component means).
  B2  The state no longer gets swamped by the pose density: a static sample
      with the *wrong* state but the right pose is rejected.
  B3  ``containment_score`` is normalized by the self-overlap, so a
      distribution always contains itself with score 1.0 (a 2-component model
      used to score only ~0.5 against itself and could never pass 0.9).
  B4  ``secure_mix_parameters`` pads up instead of crashing when the fitted
      categorical has more outcomes than ``cfg.n_states``; ``Condition`` raises
      an actionable error for that config mismatch.
  B5  Model selection computes BIC on ``model_value(values)`` (used to raise
      ``ValueError`` for entities with ``fit_rotation=False``).
  W2  ``comp_feature`` state scores use the same ``[-10, +10]`` scale as
      ``gnn_format`` instead of log-probabilities in ``[-34, 0]``.
  W5  ``best_sample`` handles disjoint categorical supports deterministically
      instead of returning an arbitrary ``argmax(0)`` state.
  G1  ``score_single`` validity is a pose gate (sigma / Mahalanobis deviation
      ``<= z_threshold``) AND a hard state-equality gate (observed state must
      equal the best component's most likely state — not tunable).
      ``sigma_deviation`` reports the z value so ``z_threshold`` can be tuned
      on real data.
  G2  ``ConPair.change_scores`` reports per entity how many pooled stds the
      post (goal) distribution mean is away from the pre mean ("mean std"),
      i.e. whether the skill actively changes the entity or only uses it as an
      anchor.
  G3  ``Graph.update_nodes`` keeps the start value for anchor entities
      (change score < ``ANCHOR_CHANGE_THRESHOLD``) instead of preferring the
      goal value.

Run from the repo root in the ``hecarim`` conda env::

    python scripts/test_02_scoring.py

Exits with status 1 if any check fails.
"""

import math
import os
import sys
import tempfile
import warnings
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplcache_heca_tests")
)

import numpy as np
from stepmix import StepMix

_REPO = Path(__file__).resolve().parents[1]
for _p in (str(_REPO), str(_REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from heca.data.entity import Entity
from heca.data.data import DCEntity, DCScene
from heca.data.static import StaticEntity
from heca.data.free import FreeEntity
from heca.conditions.condition import Condition
from heca.conditions.pair import ConPair
from heca.graphs.graph import Graph
from heca.graphs.node import OptionNode

RESULTS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record a check and print PASS/FAIL."""
    RESULTS.append((name, bool(ok), detail))
    print(
        f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else "")
    )


def fit_model(entity: Entity, data: np.ndarray, n_components: int) -> StepMix:
    model = StepMix(
        n_components=n_components,
        measurement=entity.measurement,  # type: ignore
        random_state=0,
        verbose=False,
        progress_bar=0,
    )
    model.fit(entity.model_value(data))
    return model


def static_entity(**overrides) -> Entity:
    cfg = dict(n_states=2, max_fit_components=2, threshold=0.9)
    cfg.update(overrides)
    return StaticEntity(cfg=StaticEntity.Config(**cfg))


def free_entity(**overrides) -> Entity:
    cfg = dict(n_states=2, max_fit_components=2, threshold=0.4)
    cfg.update(overrides)
    return FreeEntity(cfg=FreeEntity.Config(**cfg))


def static_data() -> np.ndarray:
    """Two one-hot states at distinct poses (forces a 2-component fit)."""
    return np.vstack(
        [
            np.tile([0.0, 0.0, 0.0, 0, 0, 0, 0.0], (50, 1)),
            np.tile([0.5, 0.0, 0.0, 0, 0, 0, 1.0], (50, 1)),
        ]
    )


def free_data(n: int = 300) -> np.ndarray:
    """Two well-separated pose clusters, each with a one-hot state."""
    rng = np.random.default_rng(0)
    pose = np.vstack(
        [
            rng.normal([0.0, 0.0, 0.0], 0.3, (n, 3)),
            rng.normal([1.0, 0.0, 0.0], 0.3, (n, 3)),
        ]
    )
    rot = rng.normal([0.0, 0.0, 0.0], 0.1, (2 * n, 3))
    ste = np.vstack([np.zeros((n, 1)), np.ones((n, 1))])
    return np.hstack([pose, rot, ste])


def legacy_score_density(entity: Entity, sample: np.ndarray, up: dict) -> float:
    """The pre-fix ``score_single``: exp of the raw mixture log-density.

    Kept here as a reference so the before/after table shows the change.
    """
    eps = 1e-15
    sample = entity.model_value(sample)
    p = entity.secure_mix_parameters(up)
    pose = sample[:-1]
    state = int(sample[-1])
    best = -np.inf
    for k in range(len(p["weights"])):
        mu = p["measurement"]["pose"]["means"][k]
        var = p["measurement"]["pose"]["covariances"][k]
        pis = p["measurement"]["state"]["pis"][k]
        log_gauss = -0.5 * np.sum(np.log(2 * np.pi * var) + (pose - mu) ** 2 / var)
        state_prob = pis[state] if state < len(pis) else eps
        log_cat = np.log(np.clip(state_prob, eps, 1.0))
        s = np.log(p["weights"][k]) + log_gauss + log_cat
        if s > best:
            best = s
    return float(np.exp(best))


def test_score_single_normalized_and_state_matters():
    print("\n== B1/B2: score_single is normalized [0,1] and the state matters ==")
    entity = static_entity()
    up = fit_model(entity, static_data(), 2).get_parameters()

    s_exact, v_exact = entity.score_single(np.array([0.0, 0, 0, 0, 0, 0, 0.0]), up)
    s_wrong_state, v_wrong_state = entity.score_single(
        np.array([0.0, 0, 0, 0, 0, 0, 1.0]), up
    )
    s_off, v_off = entity.score_single(np.array([0.01, 0, 0, 0, 0, 0, 0.0]), up)
    s_other, v_other = entity.score_single(np.array([0.5, 0, 0, 0, 0, 0, 1.0]), up)

    check(
        "static: exact match scores ~1.0 (was ~2e15)",
        abs(s_exact - 1.0) < 0.01,
        f"score={s_exact:.6f}",
    )
    check(
        "static: wrong state rejected (state gate)",
        not v_wrong_state,
        f"score={s_wrong_state:.3e}, valid={v_wrong_state}",
    )
    check(
        "static: 0.01 off the pose rejected (pose gate)",
        s_off < 0.01 and not v_off,
        f"score={s_off:.3e}, valid={v_off}",
    )
    check(
        "static: other component's mode accepted",
        s_other > 0.99 and v_other,
        f"score={s_other:.6f}, valid={v_other}",
    )
    check(
        "all static scores within [0, 1]",
        all(0.0 <= s <= 1.0 for s in [s_exact, s_wrong_state, s_off, s_other]),
    )

    # Free entity: comparable scale (was ~49 at the mean).
    fent = free_entity()
    fup = fit_model(fent, free_data(), 2).get_parameters()
    fs_mean, fv_mean = fent.score_single(np.array([0.0, 0, 0, 0, 0, 0, 0.0]), fup)
    # [0, 0.9, 0] is 3 sigma in y from *every* component (both are at y=0).
    fs_3sigma, fv_3sigma = fent.score_single(np.array([0.0, 0.9, 0, 0, 0, 0, 0.0]), fup)
    check(
        "free: at the component mean ~1.0",
        fs_mean > 0.9 and fv_mean,
        f"score={fs_mean:.4f}, valid={fv_mean}",
    )
    check(
        "free: 3sigma away rejected (pose gate)",
        fs_3sigma < 0.1 and not fv_3sigma,
        f"score={fs_3sigma:.4f}, valid={fv_3sigma}",
    )
    check("comparable scale across entity types", max(s_exact, fs_mean) <= 1.0)

    print("\n  before/after (same fitted models):")
    print(
        f"    {'sample':<30}{'z (sigma)':>10}{'old density':>14}{'new score':>12}"
        f"{'valid(old)':>12}{'valid(new)':>12}"
    )
    rows = [
        ("static exact (pose0, state0)", [0.0, 0, 0, 0, 0, 0, 0.0]),
        ("static wrong state (pose0, state1)", [0.0, 0, 0, 0, 0, 0, 1.0]),
        ("static 0.01 off pose", [0.01, 0, 0, 0, 0, 0, 0.0]),
        ("free at comp0 mean", [0.0, 0, 0, 0, 0, 0, 0.0]),
        ("free 3sigma off (y=0.9)", [0.0, 0.9, 0, 0, 0, 0, 0.0]),
    ]
    up_f = fup
    for label, sample in rows:
        ent, up_i = (entity, up) if label.startswith("static") else (fent, up_f)
        old = legacy_score_density(ent, np.array(sample), up_i)
        new, valid_new = ent.score_single(np.array(sample), up_i)
        valid_old = old >= ent.cfg.threshold
        z = ent.sigma_deviation(np.array(sample), up_i)
        print(
            f"    {label:<30}{z:>10.3f}{old:>14.3e}{new:>12.4f}"
            f"{str(valid_old):>12}{str(valid_new):>12}"
        )


def test_two_gate_thresholds():
    print("\n== G1: pose gate (z_threshold) + hard state equality ==")
    # Explicit pose gate: z_threshold = 1.0 sigma.
    entity = static_entity(z_threshold=1.0)
    up = fit_model(entity, static_data(), 2).get_parameters()

    # Build pose offsets from the fitted sigma so the test is robust to the
    # actual (floor-clamped) variances.
    sigma_x = float(math.sqrt(up["measurement"]["pose"]["covariances"][0, 0]))
    sample_at = lambda z: np.array([z * sigma_x, 0, 0, 0, 0, 0, 0.0])

    z_half = entity.sigma_deviation(sample_at(0.5), up)
    z_two = entity.sigma_deviation(sample_at(2.0), up)
    check(
        "sigma_deviation reports ~0.5 sigma",
        abs(z_half - 0.5) < 0.05,
        f"z={z_half:.3f}",
    )
    check(
        "sigma_deviation reports ~2.0 sigma", abs(z_two - 2.0) < 0.05, f"z={z_two:.3f}"
    )

    _, v_half = entity.score_single(sample_at(0.5), up)
    _, v_two = entity.score_single(sample_at(2.0), up)
    check("pose gate: 0.5 sigma passes with z_threshold=1.0", v_half)
    check("pose gate: 2.0 sigma fails with z_threshold=1.0", not v_two)

    # State gate: plain equality — same pose, wrong state is always rejected.
    _, v_bad_state = entity.score_single(np.array([0.0, 0, 0, 0, 0, 0, 1.0]), up)
    check("state gate: wrong state rejected (hard equality)", not v_bad_state)
    _, v_good_state = entity.score_single(np.array([0.0, 0, 0, 0, 0, 0, 0.0]), up)
    check("state gate: matching state accepted", v_good_state)

    # Default derivation: z_threshold=None -> sqrt(-2*ln(threshold)); the pose
    # gate then reproduces the legacy score>=threshold decision exactly.
    legacy_ent = static_entity()  # threshold=0.9, z_threshold=None
    z_derived = legacy_ent._effective_z_threshold()
    check(
        "derived z_threshold = sqrt(-2*ln(0.9)) ~ 0.459",
        abs(z_derived - math.sqrt(-2.0 * math.log(0.9))) < 1e-9,
        f"z_thresh={z_derived:.4f}",
    )
    for z_off in (0.3, 0.5, 0.7):  # crosses the derived 0.459 cutoff
        sample = sample_at(z_off)
        z = legacy_ent.sigma_deviation(sample, up)
        new_valid = legacy_ent.score_single(sample, up)[1]
        old_valid = math.exp(-0.5 * z * z) >= legacy_ent.cfg.threshold
        check(
            f"backward compat at z={z:.2f}: gate == exp(-0.5 z^2) >= threshold",
            new_valid == old_valid,
            f"new={new_valid}, old={old_valid}",
        )


def test_pair_change_scores():
    print("\n== G2: ConPair.change_scores (anchor vs actively changed) ==")
    ent = static_entity()
    rng = np.random.default_rng(0)
    aa = np.zeros((100, 3))
    ste = np.zeros((100, 1))
    pos_pre = rng.normal([0.1, -0.2, 0.3], 0.01, (100, 3))

    pre_data = {
        "anchor": np.hstack([pos_pre.copy(), aa.copy(), ste.copy()]),
        "moved": np.hstack([pos_pre.copy(), aa.copy(), ste.copy()]),
    }
    post_data = {
        # Anchor: identical to pre (only used for orientation).
        "anchor": np.hstack([pos_pre.copy(), aa.copy(), ste.copy()]),
        # Moved: post mean shifted by 0.3 in x (~30 pooled stds).
        "moved": np.hstack(
            [rng.normal([0.4, -0.2, 0.3], 0.01, (100, 3)), aa.copy(), ste.copy()]
        ),
    }
    entities = {"anchor": ent, "moved": ent}
    pair = ConPair.make("t", pre_data, post_data, entities)
    scores = pair.change_scores
    check(
        "anchor entity change score ~0",
        scores["anchor"] < 0.5,
        f"score={scores['anchor']:.4f}",
    )
    check(
        "changed entity change score large",
        scores["moved"] > 2.0,
        f"score={scores['moved']:.3f}",
    )


def test_update_nodes_anchor_behavior():
    print("\n== G3: update_nodes keeps start for anchors, goal for changed ==")
    ent = static_entity()
    rng = np.random.default_rng(1)

    def scene_values(mean):
        return np.hstack(
            [rng.normal(mean, 0.01, (100, 3)), np.zeros((100, 3)), np.zeros((100, 1))]
        )

    pre_data = {
        "anchor": scene_values([0.1, -0.2, 0.3]),
        "moved": scene_values([0.1, -0.2, 0.3]),
    }
    post_data = {
        "anchor": scene_values([0.1, -0.2, 0.3]),  # unchanged -> anchor
        "moved": scene_values([0.4, -0.2, 0.3]),  # shifted -> changed
    }
    entities = {"anchor": ent, "moved": ent}
    pair = ConPair.make("t", pre_data, post_data, entities)

    graph = Graph(entities=entities)
    pre_comp = graph.set_comps("t", pair.pre)
    post_comp = graph.set_comps("t", pair.post)
    pre_src = graph.set_precon("t", pair.pre, pre_comp)
    post_src = graph.set_postcon(
        "t", pair.post, post_comp, pre_src, change_scores=pair.change_scores
    )
    graph.ns_option.add(
        "opt_t", OptionNode(model=None, sources={src for src in post_src.values()})
    )
    graph.es_stepmix.edges_from_sets(graph.ns_entity, graph.ns_entity)
    graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option)
    graph.es_tapas.edges_from_sets(graph.ns_entity, graph.ns_entity)

    def dc(v):
        v = np.asarray(v, dtype=np.float32)
        return DCEntity(value=v, feature=ent.gnn_format(v))

    start = DCScene(
        {
            "anchor": dc([0.1, -0.2, 0.3, 0, 0, 0, 0.0]),
            "moved": dc([0.1, -0.2, 0.3, 0, 0, 0, 0.0]),
        }
    )
    # The goal's anchor pose is slightly different (but still within the post
    # model's z_threshold) so that without the anchor branch update_nodes
    # would adopt the goal value instead of the start value.
    goal = DCScene(
        {
            "anchor": dc([0.1015, -0.2, 0.3, 0, 0, 0, 0.0]),
            "moved": dc([0.4, -0.2, 0.3, 0, 0, 0, 0.0]),
        }
    )
    graph.set_goal(goal)
    graph.set_start(start)  # triggers update_nodes

    anchor_node = graph.ns_entity.get_by_key("post_anchort")
    moved_node = graph.ns_entity.get_by_key("post_movedt")
    check(
        "anchor post-node keeps the start value (ignores goal)",
        np.allclose(anchor_node.data.value, start["anchor"].value),
        f"value={anchor_node.data.value}",
    )
    check(
        "changed post-node adopts the goal value",
        np.allclose(moved_node.data.value, goal["moved"].value),
        f"value={moved_node.data.value}",
    )


def test_containment_self_and_rejection():
    print("\n== B3: containment_score normalized (self = 1.0), rejects mismatches ==")
    entity = static_entity()
    up = fit_model(entity, static_data(), 2).get_parameters()
    s_self = entity.containment_score(up, up)
    check(
        "static: self-containment = 1.0 (was ~0.5)",
        abs(s_self - 1.0) < 1e-3,
        f"score={s_self:.6f}",
    )

    # Model whose pose is shifted by 0.01.
    Xb = static_data().copy()
    Xb[:, :3] += 0.01
    upb = fit_model(entity, Xb, 2).get_parameters()
    s_shift = entity.containment_score(up, upb)
    check("static: shifted pose rejected", s_shift < 0.01, f"score={s_shift:.3e}")

    # Model whose state labels are swapped (disjoint categorical supports).
    Xs = static_data().copy()
    Xs[:, -1] = 1.0 - Xs[:, -1]
    ups = fit_model(entity, Xs, 2).get_parameters()
    s_disjoint = entity.containment_score(up, ups)
    check(
        "static: disjoint states rejected", s_disjoint < 0.1, f"score={s_disjoint:.4f}"
    )

    fent = free_entity()
    fup = fit_model(fent, free_data(), 2).get_parameters()
    s_free_self = fent.containment_score(fup, fup)
    check(
        "free: self-containment = 1.0 (was ~0.8)",
        abs(s_free_self - 1.0) < 1e-3,
        f"score={s_free_self:.6f}",
    )


def test_make_subgoal_connections():
    print(
        "\n== B3: make_subgoal connects identical static conditions, rejects disjoint =="
    )
    ent = static_entity()
    data = {"obj": static_data()}
    entities = {"obj": ent}

    np.random.seed(0)
    pre = Condition("pre", data, entities)
    np.random.seed(0)
    post = Condition("post", data, entities)
    sub = post.make_subgoal(pre)
    check(
        "identical static conditions connect (was None)",
        sub is not None,
        f"subgoal={sub}",
    )
    if sub is not None:
        score, _ = sub["obj"]
        check(
            "identical static conditions score above 0.9",
            score >= 0.9,
            f"score={score:.4f}",
        )

    data_s = {"obj": static_data().copy()}
    data_s["obj"][:, -1] = 1.0 - data_s["obj"][:, -1]
    np.random.seed(0)
    pre_s = Condition("pre", data_s, entities)
    sub_s = post.make_subgoal(pre_s)
    check("disjoint static states do NOT connect", sub_s is None, f"subgoal={sub_s}")


def test_secure_mix_no_crash_on_more_states():
    print("\n== B4: secure_mix_parameters pads up instead of crashing ==")
    ent = static_entity(n_states=1)  # deliberately too few states
    up = fit_model(ent, static_data(), 2).get_parameters()  # data has 2 outcomes
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        p = ent.secure_mix_parameters(up)
        s, _ = ent.score_single(np.array([0.0, 0, 0, 0, 0, 0, 0.0]), p)
    pis = p["measurement"]["state"]["pis"]
    check(
        "no crash; categorical padded to 2 outcomes",
        pis.shape[1] == 2,
        f"shape={pis.shape}",
    )
    check(
        "warning emitted about the n_states mismatch",
        any("n_states" in str(w.message) for w in caught),
    )
    check(
        "scoring still works with the expanded states",
        0.0 <= s <= 1.0,
        f"score={s:.4f}",
    )


def test_fit_model_bic_and_validation():
    print("\n== B5: BIC on model_value; n_states validation ==")
    # fit_rotation=False: BIC must use the transformed data (used to raise).
    ent = static_entity(fit_rotation=False)
    X = static_data()  # [pos, aa, ste] -> model sees [pos, ste]
    cond = Condition("pre", {"obj": X}, {"obj": ent})
    check(
        "Condition fits with fit_rotation=False",
        "obj" in cond.models,
        "BIC computed on model_value(values)",
    )

    # Fitted outcomes > cfg.n_states -> actionable error instead of a
    # confusing broadcast crash deep inside scoring.
    ent2 = static_entity(n_states=1, max_fit_components=1)
    try:
        Condition("pre", {"obj": X}, {"obj": ent2})
        check(
            "outcomes>n_states raises actionable ValueError",
            False,
            "no exception raised",
        )
    except ValueError as exc:
        check(
            "outcomes>n_states raises actionable ValueError",
            "n_states" in str(exc),
            str(exc)[:90],
        )


def test_comp_feature_scale_and_best_sample():
    print("\n== W2/W5: comp_feature scale + best_sample disjoint fallback ==")
    entity = static_entity()
    up = fit_model(entity, static_data(), 2).get_parameters()

    feats = entity.comp_feature(up)
    state_region = feats[:, : entity.cfg.n_states]
    check(
        "comp_feature state scores within [-10, +10]",
        float(state_region.min()) >= -10.0 - 1e-3
        and float(state_region.max()) <= 10.0 + 1e-3,
        f"min={state_region.min():.2f}, max={state_region.max():.2f}",
    )
    check(
        "one-hot state encoded as +10 (gnn_format convention)",
        np.allclose(np.max(state_region, axis=1), 10.0),
    )

    # best_sample with disjoint one-hot state supports: no crash, valid state.
    Xs = static_data().copy()
    Xs[:, -1] = 1.0 - Xs[:, -1]
    ups = fit_model(entity, Xs, 2).get_parameters()
    val = entity.best_sample(up, ups)
    check("best_sample handles disjoint states", int(val[-1]) in (0, 1), f"value={val}")

    # gnn_format guard: out-of-range state id must not write outside the slice.
    feat = entity.gnn_format(np.array([0.0, 0, 0, 0, 0, 0, 5.0]))  # state 5, n_states=2
    check(
        "gnn_format guards out-of-range state",
        np.all(feat[:2] == -10.0) and float(feat[5]) == 0.0,
        f"feat[:2]={feat[:2]}, feat[5]={feat[5]}",
    )


def main():
    print("heca scoring regression tests")
    print(f"repo: {_REPO}\n")

    tests = [
        test_score_single_normalized_and_state_matters,
        test_two_gate_thresholds,
        test_pair_change_scores,
        test_update_nodes_anchor_behavior,
        test_containment_self_and_rejection,
        test_make_subgoal_connections,
        test_secure_mix_no_crash_on_more_states,
        test_fit_model_bic_and_validation,
        test_comp_feature_scale_and_best_sample,
    ]
    for t in tests:
        t()

    failed = [r for r in RESULTS if not r[1]]
    print("\n" + "=" * 64)
    print(f"Summary: {len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    if failed:
        print("Failed checks:")
        for name, _, detail in failed:
            print(f"  - {name}: {detail}")
        sys.exit(1)

    print("\nNote: score_single validity is the AND of two gates:")
    print(
        "      pose gate : sigma deviation z <= z_threshold  (None -> sqrt(-2*ln(threshold)))"
    )
    print("      state gate: hard equality — observed state must equal the best")
    print("                   component's most likely state (not tunable).")
    print("      The returned [0,1] score is kept for diagnostics/ranking only.")
    print("      containment_score (make_subgoal) still uses cfg.threshold")
    print("      (0.9 static, 0.4 free, 0.6 prismatic/revolute).")
    print("All checks passed.")


if __name__ == "__main__":
    main()
