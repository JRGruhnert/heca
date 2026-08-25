import argparse
import os
import sys
from pathlib import Path

# Make ``conf`` and ``scripts.common`` importable when run directly as
# ``python scripts/a05_view_model_reliability.py``.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.data.entity import Entity
from heca.experts.expert import ExpertModel
from heca.experts.tapas import TapasExpert
from heca.misc import logger
from heca.scenes.ogbench.scene import OGScene
from heca.scenes.scene import Scene, SceneFeedback

from scripts.common.args import add_scene_argument
from scripts.common.scenes import find_scene_config, find_scene_models


def entity_str(label: str, dc: DCEntity) -> str:
    """Compact per-type value formatting for display."""
    if any(p in label for p in ("cube", "lid", "peg")):
        return f"pos={np.round(np.asarray(dc.pos, dtype=float), 4).tolist()}"
    if any(p in label for p in ("button", "box", "shelf")):
        return f"ste={int(dc.ste)}"
    if any(p in label for p in ("faucet", "lever")):
        return f"ang={float(np.arctan2(dc.ext[0], dc.ext[1])):+.4f}"
    if any(p in label for p in ("drawer", "window", "slider")):
        return f"ext={float(dc.ext[0]):+.4f}"
    return f"value={np.round(np.asarray(dc.value, dtype=float), 4).tolist()}"


def fmt_value(v):
    """Round arbitrary info_dict values (arrays / tuples / scalars) for print."""
    if isinstance(v, np.ndarray):
        return np.round(np.asarray(v, dtype=float), 4).tolist()
    if isinstance(v, tuple):
        return tuple(fmt_value(x) for x in v)
    return v


def value_diff(a, b) -> float:
    return float(
        np.linalg.norm(
            np.asarray(a.value, dtype=float) - np.asarray(b.value, dtype=float)
        )
    )


def print_scene(title: str, scene: DCScene, labels: list[str] | None = None):
    print(f"  {title}:")
    for label, entity in scene.entities():
        if labels is not None and label not in labels:
            continue
        print(f"    {label:<22} {entity_str(label, entity)}")


def mode_dc(entity: Entity, model) -> DCEntity:
    """Most probable value (best-component mean + mode state) of a fitted
    StepMix model, as a displayable DCEntity."""
    p = model.get_parameters()
    k = int(np.argmax(p["weights"]))
    pose = p["measurement"]["pose"]["means"][k]
    state = int(np.argmax(p["measurement"]["state"]["pis"][k]))
    value = np.concatenate([np.asarray(pose, dtype=float), [float(state)]])
    return DCEntity(value=value, feature=np.zeros(0))


def precondition_status(agent: TapasExpert, x: DCScene) -> tuple[bool, list[str]]:
    """Is the current scene ``x`` a valid start for the agent's pre-condition?

    Per entity: ``score_single`` against the fitted pre-model (pose gate +
    hard state gate). Returns ``(all_valid, blocked_entities)``.
    """
    blocked = []
    for label, entity in agent.entities.items():
        _, valid = entity.score_single(
            x.get(label).value,
            agent.conditions.pre.models[label].get_parameters(),
        )
        if not valid:
            blocked.append(label)
    return len(blocked) == 0, blocked


def agent_expectation(agent: TapasExpert) -> str:
    """'a -> b' of the moving (target) entities, from the pre/post condition
    modes, e.g. ``faucet0: ang=+0.25 -> ang=-1.33`` (matches the model tag)."""
    pre, post = agent.conditions.pre, agent.conditions.post
    parts = []
    for label in sorted(agent.conditions.target_entities):
        ent = pre.entities[label]
        parts.append(
            f"{label}: {entity_str(label, mode_dc(ent, pre.models[label]))}"
            f" -> {entity_str(label, mode_dc(ent, post.models[label]))}\n"
        )
    return "  ".join(parts)


def run_virtual_step(
    scene: OGScene, agent: TapasExpert, x: DCScene, y: DCScene
) -> tuple[DCScene, SceneFeedback]:
    print("\n" + "=" * 78)
    print(f"VIRTUAL STEP (roundtrip) — agent: {agent.cfg.tag}")
    print("=" * 78)

    change_scores = agent.conditions.change_scores
    targets = agent.conditions.target_entities
    anchors = agent.conditions.anchor_entities

    print(f"\nchange scores (anchor threshold = {Entity.ANCHOR_THRESHOLD}):")
    for label in sorted(agent.entities.keys()):
        score = change_scores.get(label, float("nan"))
        kind = "TARGET" if label in targets else "ANCHOR"
        print(f"    {label:<22} change={score:>8.3f}  [{kind}]")

    print("\ngate — start value fits pre-model (score_single):")
    blocked_by = []
    for label, entity in agent.entities.items():
        score, valid = entity.score_single(
            x.get(label).value,
            agent.conditions.pre.models[label].get_parameters(),
        )
        print(f"    {label:<22} valid={valid!s:<5} score={score:.4f}")
        if not valid:
            blocked_by.append(label)

    z, fb = agent._act_virt(x, y)

    if blocked_by:
        print(
            f"\n-> step BLOCKED by: {blocked_by} "
            "(starting conditions not met; step_scene NOT called)."
        )
        return z, fb

    sent: dict = {}
    for label in agent.conditions.target_entities:
        sent.update(
            scene.entities[label].env_state_value(
                label, y, unnormalize_pos=scene.unnormalize_position
            )
        )
    print("\ninfo_dict sent to step_scene (mirrors _step_virt):")
    for key, value in sent.items():
        print(f"    {key:<28} = {fmt_value(value)}")
    print(f"    -> targets sent    : {sorted(targets)}")
    print(f"    -> anchors left out (unchanged): {sorted(anchors)}")

    print(
        f"\nstep_scene feedback: terminal={fb.terminal} "
        f"reward={fb.reward:.4f} truncated={fb.truncated}"
    )
    print("\nresult z (per entity, x -> z):")
    for label in sorted(agent.entities.keys()):
        vx, vz = x.get(label), z.get(label)
        print(f"    {label:<22} {entity_str(label, vx)}  ->  {entity_str(label, vz)}")

    print("\nroundtrip check (per entity):")
    for label in sorted(agent.entities.keys()):
        dxz = value_diff(x.get(label), z.get(label))
        dzy = value_diff(z.get(label), y.get(label))
        if label in targets:
            status = "OK" if dzy < 1e-3 else "MISMATCH"
            note = f"|z-y|={dzy:.5f} (target: should be ~0)"
        else:
            status = "OK" if dxz < 1e-3 else "CHANGED?!"
            note = f"x->z={dxz:.5f} (anchor: must stay at x)"
        print(f"    {label:<22} [{status:<8}] {note}")

    targets_clean = all(value_diff(z.get(l), y.get(l)) < 1e-3 for l in targets)
    anchors_unchanged = all(value_diff(z.get(l), x.get(l)) < 1e-3 for l in anchors)
    print(
        f"\nROUNDTRIP SUMMARY: targets match y: {targets_clean} | "
        f"anchors unchanged: {anchors_unchanged}"
    )
    return z, fb


def run_real_rollout(
    scene: OGScene, agent: TapasExpert, x: DCScene, y: DCScene
) -> tuple[DCScene, SceneFeedback]:
    """Execute the real TAPAS policy from x toward y (physics steps)."""
    print("\n" + "=" * 78)
    print(f"REAL ROLLOUT (TAPAS policy) — agent: {agent.cfg.tag}")
    print("=" * 78)
    print_scene("goal y", y, labels=list(agent.entities.keys()))
    print()

    agent.policy.reset_episode()
    xt = agent.tapas_td(x, y)
    z = x
    fb = SceneFeedback(terminal=True, reward=0.0, truncated=False)
    step = 0

    def rollout_step(action: np.ndarray):
        nonlocal z, fb
        tdscene, tdimage, fb = scene.step(action)
        z = agent.make_scene(tdscene, tdimage)

    try:
        if agent.cfg.policy.return_full_batch:
            predictions = agent.make_batch_prediction(xt)
            if predictions is None:
                print("-> prediction failed (None), aborting rollout.")
                return x, SceneFeedback(terminal=True, reward=0.0, truncated=False)
            while not predictions.is_finished:
                pred = predictions.step()
                action = np.concatenate((pred.ee, pred.gripper))  # type: ignore
                rollout_step(action)
                step += 1
                print(
                    f"    step {step:>4d}: |action|={np.linalg.norm(action):.3f} "
                    f"fb=(t={fb.terminal}, r={fb.reward:.3f})"
                )
        else:
            while not (pred := agent.make_prediction(xt))[1]:
                action, _ = pred
                if action is None:
                    print("-> prediction failed (None), aborting rollout.")
                    return x, SceneFeedback(terminal=True, reward=0.0, truncated=False)
                rollout_step(action)
                step += 1
                xt = agent.tapas_td(z, y)
                print(
                    f"    step {step:>4d}: |action|={np.linalg.norm(action):.3f} "
                    f"fb=(t={fb.terminal}, r={fb.reward:.3f})"
                )
    except KeyboardInterrupt:
        print("\n-> rollout interrupted by user.")

    print(f"\nrollout finished after {step} steps.")
    print(
        f"feedback: terminal={fb.terminal} reward={fb.reward:.4f} "
        f"truncated={fb.truncated}"
    )
    print("\nresult z (per entity, x -> z):")
    for label in sorted(agent.entities.keys()):
        vx, vz = x.get(label), z.get(label)
        print(f"    {label:<22} {entity_str(label, vx)}  ->  {entity_str(label, vz)}")
    return z, fb


def sample_task(scene: OGScene) -> tuple[DCScene, DCScene]:
    (s_scene, _), (g_scene, _) = scene.sample_task()
    return s_scene, g_scene


def print_menu(agents: list[ExpertModel], x: DCScene):
    print("\n" + "-" * 78)
    print("Options (pre = precondition fulfilled by the current scene x):")
    for i, agent in enumerate(agents):
        row = f"  [v{i}]/[r{i}]  {agent.cfg.tag:<24}"
        try:
            ok, blocked = precondition_status(agent, x)
            flag = "TRUE " if ok else "FALSE"
            note = "" if ok else f" (blocked by: {', '.join(blocked)})"
            exp = agent_expectation(agent)
        except Exception as exc:
            flag, note, exp = "n/a  ", f" (conditions unavailable: {exc})", ""
        print(f"{row} pre={flag}{note}")
        if exp:
            print(f"          expects: {exp}")
    print("  [0]   reset / sample a new episode")
    print("  [q]   quit")
    print("-" * 78)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser, default="scene1")
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth observations (default: true).",
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Enable the passive viewer.",
    )
    args = parser.parse_args()

    scene_cfg = find_scene_config(args.scene)
    scene = Scene.get(scene_cfg, auto_load=False)
    assert isinstance(scene, OGScene), "Only OGScene supported."
    if args.vis:
        scene.cfg.visualize = True

    agents: list[ExpertModel] = []
    for agent_cfg in find_scene_models(args.scene):
        agent = ExpertModel.get(agent_cfg, auto_load=False)
        agent.use_gt(args.use_gt)
        agent.load()
        agents.append(agent)

    # Warm up the conditions once (fits StepMix / loads conditions.joblib).
    for agent in agents:
        try:
            _ = agent.conditions.target_entities
        except Exception as exc:  # keep going; the virtual step will report it
            logger.warning(f"[{agent.cfg.tag}] conditions unavailable: {exc}")

    try:
        quitting = False
        episode = 0
        x, y = sample_task(scene)
        while True:
            episode += 1
            print("\n" + "=" * 78)
            print(
                f"EPISODE {episode} — scene: {scene_cfg.tag} " f"(use_gt={args.use_gt})"
            )
            print("=" * 78)
            print_scene("current scene x", x)
            print_scene("goal y", y)
            print_menu(agents, x)

            choice = input("> ").strip().lower()
            if choice in ("q", "quit", "exit"):
                quitting = True
                break
            if choice in ("0", "reset", ""):
                x, y = sample_task(scene)
                continue

            kind, idx = choice[0], choice[1:]
            if kind not in ("v", "r") or not idx.isdigit():
                print(f"  unknown command: {choice!r}")
                continue
            i = int(idx)
            if i >= len(agents):
                print(f"  no agent with index {i}")
                continue
            agent = agents[i]
            assert isinstance(agent, TapasExpert)
            if kind == "v":
                z, fb = run_virtual_step(scene, agent, x, y)
            else:
                z, fb = run_real_rollout(scene, agent, x, y)
            x = z  # chain: the env has actually moved
            if fb.terminal or fb.truncated:
                print("\n  (episode ended by feedback; sampling a new task)")
                x, y = sample_task(scene)
    except (EOFError, KeyboardInterrupt):
        quitting = True
        print("\nbye.")
    finally:
        scene.close()
        if quitting:
            # Intended quit: skip interpreter teardown (torch / mujoco / GLFW /
            # h5py atexit handlers) — this tool saves nothing, so a hard exit
            # returns to the shell instantly. Unexpected exceptions are NOT
            # swallowed: they fall through and print their traceback.
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)


if __name__ == "__main__":
    main()
