import argparse

import matplotlib

matplotlib.use("Agg")

import numpy as np

from heca.data.data import DCScene
from heca.experts.expert import ExpertModel
from heca.scenes.ogbench.scene import OGScene
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

N_RANDOM_ROLLOUTS = 60
MAX_PLAN_STEPS = 12
PLAN_LOOKAHEAD = 2

# Env success tolerances (world metres; DCScene positions are scaled by 10).
FREE_TOL_M = 0.04
FAUCET_TOL_RAD = 0.15


def fmt(v):
    if isinstance(v, np.ndarray):
        return np.round(np.asarray(v, dtype=float), 4).tolist()
    return v


# ---------------------------------------------------------------- entity utils


def entity_kind(entity) -> str:
    t = entity.cfg.type_id
    return {0: "free", 1: "static", 2: "prismatic", 3: "revolute"}.get(t, "other")


def mismatched(x: DCScene, y: DCScene, scene) -> dict[str, float]:
    """Goal entities whose current value differs from the goal value.

    Returns {label: approx mismatch magnitude} using the same success criteria
    the ogbench env uses (button state, faucet angle tolerance, free-body pos).
    """
    out = {}
    for label, entity in scene.entities.items():
        kind = entity_kind(entity)
        vx, vy = x.get(label).value, y.get(label).value
        if kind == "static":
            sx, sy = int(round(vx[-1])), int(round(vy[-1]))
            if sx != sy:
                out[label] = float(sy - sx)
        elif kind == "revolute":
            ang_x = float(np.arctan2(vx[6], vx[7]))
            ang_y = float(np.arctan2(vy[6], vy[7]))
            d = abs(ang_x - ang_y)
            if d > FAUCET_TOL_RAD:
                out[label] = float(d)
        elif kind == "free":
            d = float(np.linalg.norm(vx[:3] - vy[:3]))
            if d > FREE_TOL_M * 10.0:
                out[label] = d
        else:
            d = float(np.linalg.norm(vx - vy))
            if d > 1e-6:
                out[label] = d
    return out


# ------------------------------------------------------------- env bookkeeping


def env_snapshot(scene) -> tuple:
    env = scene.env
    obj_recs = []
    for o in env.objects:
        rec = {}
        for attr in ("_cur_state", "_target_button_states"):
            v = getattr(o, attr, None)
            rec[attr] = v.copy() if v is not None else None
        rec["_target_val"] = getattr(o, "_target_val", None)
        obj_recs.append((o, rec))
    return env._data.qpos.copy(), env._data.qvel.copy(), obj_recs


def env_restore(scene, snap) -> None:
    env = scene.env
    qpos, qvel, obj_recs = snap
    env._data.qpos[:] = qpos
    env._data.qvel[:] = qvel
    for o, rec in obj_recs:
        for attr, val in rec.items():
            if val is None:
                continue
            if attr == "_target_val":
                setattr(o, attr, val)
            else:
                cur = getattr(o, attr)
                cur[:] = val
    env._apply_button_states()
    # refresh the cached success flag for the restored state
    env._success = bool(env._evaluate_success(env._compute_successes()))


def env_success(scene) -> bool:
    return bool(scene.env._success)


# ------------------------------------------------------------------ skill utils


def pre_valid(agent: ExpertModel, x) -> bool:
    for label, entity in agent.entities.items():
        ok = entity.score_single(
            x.get(label).value,
            agent.conditions.pre.models[label].get_parameters(),
        )
        if not ok:
            return False
    return True


def pre_fail_entities(agent: ExpertModel, x) -> list[str]:
    """Entity labels whose pre-condition fails at x ([] => pre valid)."""
    bad = []
    for label, entity in agent.entities.items():
        ok = entity.score_single(
            x.get(label).value,
            agent.conditions.pre.models[label].get_parameters(),
        )
        if not ok:
            bad.append(label)
    return bad


def agent_subgoal(x, y, agent, mode: str):
    """Subgoal scene for `agent` at current x, mirroring the training graph.

    mode="graph": target = y if inside post-condition else a post sample
                  (what the SIMPLE subgoal-mode option actually proposes)
    mode="cheat": target = y always (optimistic upper bound, ignores post cov.)
    """
    from heca.data.data import DCEntity

    sub = x.copy()
    con = agent.conditions
    for e in con.target_entities:
        if mode == "cheat" or con.post.test(e, y):
            val = y[e].value
        else:
            val = con.post.sample(e).value
        sub.set(e, DCEntity(value=val, feature=np.zeros(0)))
    return sub


def try_skill(scene, agent, x, y, mode: str):
    """Run one virtual skill; returns (z, feedback, success). Env untouched."""
    snap = env_snapshot(scene)
    try:
        if not pre_valid(agent, x):
            return None
        sub = agent_subgoal(x, y, agent, mode)
        z, fb = agent.act(x, sub)
        ok = env_success(scene) or bool(fb.terminal and fb.reward == 1.0)
        return z, fb, ok
    finally:
        env_restore(scene, snap)


def run_skill_commit(scene, agent, x, y, mode: str):
    """Commit one virtual skill (env left in the new state)."""
    sub = agent_subgoal(x, y, agent, mode)
    z, fb = agent.act(x, sub)
    return z, fb


# ------------------------------------------------------------------- solvers


def solve_oracle(scene, agents, x, y, mode: str, max_steps=MAX_PLAN_STEPS):
    """Greedy + limited lookahead. Returns (solved, n_steps, blockers, final_x)."""
    cur = x
    steps = 0
    blockers = None
    while steps < max_steps:
        if env_success(scene):
            return True, steps, None, cur
        cands = []
        for agent in agents:
            res = try_skill(scene, agent, cur, y, mode)
            if res is None:
                continue
            z, fb, ok = res
            cands.append((agent, z, ok))
        if not cands:
            return False, steps, sorted(mismatched(cur, y, scene).keys()), cur

        # lookahead: prefer candidates that lead to success, then reduce the
        # mismatch count the most, then the smallest residual distance
        best = None
        best_key = None
        for agent, z, ok in cands:
            if ok:
                return True, steps + 1, None, cur
            rem = mismatched(z, y, scene)
            score = -len(rem)
            key = (score, sum(abs(v) for v in rem.values()), agent.cfg.tag)
            if best_key is None or key > best_key:
                best_key = key
                best = (agent, z)
        # greedy commit of the best candidate
        agent, z = best
        if mismatched(z, y, scene) == mismatched(cur, y, scene):
            # no progress; try a second agent from this state (depth-2)
            improved = False
            snap = env_snapshot(scene)
            try:
                res2 = [
                    r
                    for r in (try_skill(scene, a, z, y, mode) for a in agents)
                    if r is not None
                ]
                improved = any(ok2 for _, _, ok2 in res2)
            finally:
                env_restore(scene, snap)
            if not improved:
                return (
                    False,
                    steps,
                    sorted(mismatched(cur, y, scene).keys()),
                    cur,
                )
        cur, fb = run_skill_commit(scene, agent, cur, y, mode)
        steps += 1
    return env_success(scene), steps, sorted(mismatched(cur, y, scene).keys()), cur


def solve_random(scene, agents, x, y, mode: str, max_steps=MAX_PLAN_STEPS):
    """Monte-Carlo rollouts: uniformly pick among precondition-valid skills."""
    cur = x
    steps = 0
    while steps < max_steps:
        if env_success(scene):
            return True, steps
        valid = []
        for agent in agents:
            if pre_valid(agent, cur):
                valid.append(agent)
        if not valid:
            return False, steps
        agent = valid[int(np.random.randint(len(valid)))]
        cur, fb = run_skill_commit(scene, agent, cur, y, mode)
        steps += 1
    return env_success(scene), steps


class _Found(Exception):
    pass


def solve_dfs(
    scene, agents, x, y, mode: str, max_depth: int = 14
) -> tuple[bool, int, bool]:
    """Exhaustive search over skill sequences (real env, snapshot rollback).

    Invariant: ``dfs(cur)`` is called with the real env state equal to ``cur``.
    Returns (solved, steps, cut_by_depth):
      - solved: a sequence reaching env success was found
      - cut_by_depth=False with solved=False => the search space was exhausted
        (the episode is structurally unsolvable by this skill library)
      - cut_by_depth=True with solved=False => hit the depth/transition cap
        (inconclusive)
    """
    node_budget = [400_000]

    def signature(cur) -> tuple:
        parts = []
        for label, entity in scene.entities.items():
            kind = entity_kind(entity)
            v = cur.get(label).value
            if kind == "static":
                parts.append(("s", label, int(round(v[-1]))))
            elif kind == "revolute":
                # bucket the continuous angle so the memo can prune
                ang = round(float(np.arctan2(v[6], v[7])) / 0.01) * 0.01
                parts.append(("r", label, ang))
            elif kind == "free":
                d = float(np.linalg.norm(v[:3] - y.get(label).value[:3]))
                parts.append(("f", label, d <= FREE_TOL_M * 10.0))
        return tuple(parts)

    # failed[(sig, steps)]: this state at this remaining budget was explored
    # and dead-ended. Reaching the same sig with *more* remaining budget
    # (smaller steps) must be re-explored -> exact for a min-steps search.
    failed: set = set()
    cut_by_depth = [False]
    best_found = [None]  # min steps to success

    def dfs(cur, depth, steps) -> bool:
        if env_success(scene):
            if best_found[0] is None or steps < best_found[0]:
                best_found[0] = steps
            return True
        if depth == 0:
            cut_by_depth[0] = True
            return False
        if best_found[0] is not None and steps >= best_found[0]:
            return False
        sig = signature(cur)
        if (sig, steps) in failed:
            return False
        snap_cur = env_snapshot(scene)
        ok_any = False
        try:
            for agent in agents:
                if node_budget[0] <= 0:
                    cut_by_depth[0] = True
                    break
                if not pre_valid(agent, cur):
                    continue
                env_restore(scene, snap_cur)  # env == cur again
                z, _ = run_skill_commit(scene, agent, cur, y, mode)  # cur -> z
                node_budget[0] -= 1
                if env_success(scene):
                    if best_found[0] is None or steps + 1 < best_found[0]:
                        best_found[0] = steps + 1
                    return True
                nsig = signature(z)
                if nsig == sig or (nsig, steps + 1) in failed:
                    continue
                if dfs(z, depth - 1, steps + 1):  # env back to z on return
                    ok_any = True
        finally:
            env_restore(scene, snap_cur)
        if not ok_any:
            failed.add((sig, steps))
        return ok_any

    try:
        dfs(x, max_depth, 0)
    except RecursionError:
        cut_by_depth[0] = True
    return (best_found[0] is not None), (best_found[0] or 0), cut_by_depth[0]


# ----------------------------------------------------------------------- main


def _explain_blocked(scene, agents, cur, y, stats):
    """Tally, for every still-mismatched goal entity, which skill's
    precondition keeps it unreachable at the stuck state `cur`."""
    mis = mismatched(cur, y, scene)
    for goal_e in mis:
        for a in agents:
            if goal_e not in a.conditions.target_entities:
                continue
            bad = pre_fail_entities(a, cur)
            key = f"{goal_e}<-{a.cfg.tag} blocked by {sorted(bad)}"
            stats["blocked_by"][key] = stats["blocked_by"].get(key, 0) + 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", required=True)
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--rollouts", type=int, default=N_RANDOM_ROLLOUTS)
    parser.add_argument(
        "--search",
        choices=["greedy", "dfs"],
        default="greedy",
        help="Oracle solver: greedy+lookahead or exhaustive DFS (slower).",
    )
    parser.add_argument(
        "--mode",
        choices=["cheat", "graph"],
        default="cheat",
        help="Subgoal semantics: 'cheat' teleports each target straight to the "
        "goal value; 'graph' mirrors the training SIMPLE graph (goal value if "
        "inside the skill post-condition, otherwise a post sample).",
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)
    assert isinstance(scene, OGScene)

    agents = []
    for agent_cfg in find_scene_models(args.scene):
        agent = ExpertModel.get(agent_cfg, auto_load=False)
        agent.use_gt(True)
        agent.virtual()
        agent.load()
        agents.append(agent)

    print(f"# scene={args.scene}  agents={len(agents)}")
    print(f"# agents:")
    for a in agents:
        con = a.conditions
        print(
            f"    {a.cfg.tag:<22} targets={sorted(con.target_entities)} "
            f"anchors={sorted(con.anchor_entities)}"
        )

    # per-skill: which goal entity does it cover + what are its target states
    goal_labels = sorted(scene.entities.keys())

    stats = {
        "episodes": 0,
        "n_goal_entities": [],
        "goal_entity_hist": {},
        "coverage_ok": 0,
        "pre_valid_at_start": 0,
        "oracle_cheat_ok": 0,
        "oracle_graph_ok": 0,
        "random_ok": 0,
        "random_trials": 0,
        "oracle_steps": [],
        "random_steps": [],
        "blockers": {},
        "blocked_by": {},
        "dfs_cut": [0, 0],  # [exhausted-unsolvable, cut-by-depth]
    }

    for ep in range(args.episodes):
        (x, _), (y, _) = scene.sample_task()
        stats["episodes"] += 1
        mis = mismatched(x, y, scene)
        stats["n_goal_entities"].append(len(mis))
        for e in mis:
            stats["goal_entity_hist"][e] = stats["goal_entity_hist"].get(e, 0) + 1

        # coverage: every mismatched entity covered by some skill's post
        covered = set()
        for a in agents:
            con = a.conditions
            for e in mis:
                if e in con.target_entities and con.post.test(e, y):
                    covered.add(e)
        if covered >= set(mis.keys()):
            stats["coverage_ok"] += 1

        if any(pre_valid(a, x) for a in agents):
            stats["pre_valid_at_start"] += 1

        # --- oracle on the live env ---
        if args.search == "dfs":
            solved, steps, cut = solve_dfs(scene, agents, x, y, args.mode)
            stats["oracle_cheat_ok"] += int(solved)
            stats["dfs_cut"][int(cut)] += 1
            if solved:
                stats["oracle_steps"].append(steps)
            else:
                # fall back to greedy to produce a concrete stuck state for
                # the blocker analysis (DFS restores the env after every try)
                _, _, blockers, cur = solve_oracle(scene, agents, x, y, args.mode)
                key = "|".join(blockers or ["?"])
                stats["blockers"][key] = stats["blockers"].get(key, 0) + 1
                _explain_blocked(scene, agents, cur, y, stats)
        else:
            solved, steps, blockers, cur = solve_oracle(scene, agents, x, y, args.mode)
            stats["oracle_cheat_ok"] += int(solved)
            if solved:
                stats["oracle_steps"].append(steps)
            else:
                key = "|".join(blockers or ["?"])
                stats["blockers"][key] = stats["blockers"].get(key, 0) + 1
                _explain_blocked(scene, agents, cur, y, stats)

        # --- random policy rollouts (env returns from oracle state; resample) ---
        if ep < args.rollouts:
            (x, _), (y, _) = scene.sample_task()
            ok, st = solve_random(scene, agents, x, y, "graph")
            stats["random_ok"] += int(ok)
            stats["random_trials"] += 1
            if ok:
                stats["random_steps"].append(st)

    n = stats["episodes"]
    print(f"\n## results over {n} episodes  (subgoal mode: {args.mode})")
    print(f"goal entities/episode   : {np.mean(stats['n_goal_entities']):.2f}")
    print(f"goal entity frequency   : {stats['goal_entity_hist']}")
    print(f"start pre-valid any     : {stats['pre_valid_at_start']}/{n}")
    print(f"full coverage (post)    : {stats['coverage_ok']}/{n}")
    print(
        f"oracle solve ({args.mode})  : {stats['oracle_cheat_ok']}/{n} "
        f"(mean steps {np.mean(stats['oracle_steps']) if stats['oracle_steps'] else '-'})"
    )
    if args.search == "dfs":
        print(
            f"dfs outcome on failures  : "
            f"unsolvable(exhausted)={stats['dfs_cut'][0]}, "
            f"depth-limited={stats['dfs_cut'][1]}"
        )
    print(
        f"random policy solve     : {stats['random_ok']}/{stats['random_trials']} "
        f"(mean steps {np.mean(stats['random_steps']) if stats['random_steps'] else '-'})"
    )
    print(f"blocker profile         : {stats['blockers']}")
    if stats["blocked_by"]:
        print("blocked-by detail (goal<-skill blocked by [entity]):")
        for k, v in sorted(stats["blocked_by"].items(), key=lambda kv: -kv[1]):
            print(f"    {v:>3}  {k}")


if __name__ == "__main__":
    main()
