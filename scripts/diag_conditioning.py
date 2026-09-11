"""Does the goal conditioning actually move the actor's decision?

Builds graphs for a fixed start with *different* goals (the feasible option set
only depends on the start, so the logits are element-wise comparable), and
measures how much the policy reacts to the goal:

1. argmax invariance under a goal swap, and |Δlogit| relative to the spread of
   the logits across options — a policy that ignores the goal picks the same
   option for every goal,
2. the same after removing the goal slot entirely (zeroed conditioning) and
   after removing the memory conditioning, i.e. what the conditioning is worth,
3. agreement with an explicit "effect matching" rule: pick the option whose
   effect (post-condition minus pre-condition) is most aligned with the
   current -> goal displacement. This is the bilinear interaction FiLM cannot
   represent, so the gap between the two is the headroom of an explicit
   interaction term.

Checkpoints written before the actor/critic split are remapped on load (the
previously shared modules are copied into both networks), so the finished
``test9`` runs can still be inspected.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl python scripts/diag_conditioning.py \
        --network both --tag scene0_test9-both-_gv_b --ckp ckp_500.pt --episodes 60
"""

import argparse
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import conf.networks
from heca.data.entity import Entity
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.heca_gnn.network import Network
from heca.misc import hardware
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

# Blocks that carry the same meaning in an option effect and in a canonical
# row (both write point/fitted means there); the log-std blocks differ between
# the two producers, so they are left out of the cosine.
MEAN_BLOCKS = ("state", "pos", "rot", "extra")


def mean_mask() -> np.ndarray:
    mask = np.zeros(Entity.FEATURE_DIM, dtype=bool)
    for name in MEAN_BLOCKS:
        block = Entity.LAYOUT[name]
        mask[block.mean()] = True
    return mask


def remap_legacy(state: dict[str, torch.Tensor]) -> tuple[dict, list[str]]:
    """Map a pre-split state_dict (flat keys) onto the actor/critic namespaces.

    The old network shared ``entity_encoders`` and ``state_aggregation`` and had
    one FiLM stack for both sites, so those entries are copied into both
    networks, each of which now composes its own ``EntityRowEncoder``. Unmapped
    keys are returned for reporting.
    """
    shared_prefixes = ("entity_encoders.", "state_aggregation.")
    actor_prefixes = (
        "option_encoder.",
        "condition_layer.",
        "translation_layer.",
        "summary_layer.",
        "interaction_layer.",
        "timeline_layer.",
        "option_readout.",
    )
    out: dict[str, torch.Tensor] = {}
    unmapped: list[str] = []
    for key, value in state.items():
        if key.startswith(shared_prefixes):
            out[f"actor_net.encoder.{key}"] = value
            out[f"critic_net.encoder.{key}"] = value
        elif key.startswith(actor_prefixes):
            out[f"actor_net.{key}"] = value
        elif key.startswith("state_critic."):
            out[f"critic_net.{key}"] = value
        elif key == "films.scales.actor":
            out["actor_net.films.scales.actor"] = value
        elif key == "films.scales.critic":
            out["critic_net.films.scales.critic"] = value
        elif key.startswith("films.generators.goal."):
            out[f"actor_net.{key}"] = value
            out[f"critic_net.{key}"] = value
        elif key.startswith("films.generators.memory."):
            # the critic has no memory input any more
            out[f"actor_net.{key}"] = value
        else:
            unmapped.append(key)
    return out, unmapped


def cos_to(a: np.ndarray, b: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=-1, keepdims=True) + eps)
    b = b / (np.linalg.norm(b, axis=-1, keepdims=True) + eps)
    return (a * b).sum(axis=-1)


def effect_scores(data, mask: np.ndarray) -> np.ndarray:
    """cos(effect_i, goal - current) over the mean blocks, per option."""
    can = data["canonical"]
    cur = can.x[can.cur_idx].numpy()
    goal = can.x[can.goal_idx].numpy()
    delta = (goal - cur).mean(axis=0)
    effects = data["option"].x.numpy()
    return cos_to(effects[:, mask], delta[None, mask])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="both", choices=conf.networks.NETWORK_NAMES)
    parser.add_argument("--tag", default="scene0_test9-both-_gv_b")
    parser.add_argument("--ckp", default="ckp_500.pt")
    parser.add_argument("--episodes", type=int, default=60)
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--smode", default="both")
    args = parser.parse_args()

    torch.manual_seed(0)
    np.random.seed(0)

    scene = Scene.get(find_scene_config(args.scene), auto_load=False)
    cfgs = find_scene_models(args.scene)
    for c in cfgs:
        e = ExpertModel.get(c, auto_load=False)
        e.use_gt(True)
        e.load()
        e.virtual()
    graph = Graph.generate(list(cfgs), smode=SubgoalMode(args.smode))

    net = Network.get(getattr(conf.networks, args.network))
    ckp_path = Path("data/network/standard") / args.tag / args.ckp
    raw = torch.load(ckp_path, map_location=hardware.device, weights_only=False)["network"]
    mapped, unmapped = remap_legacy(raw)
    missing, unexpected = net.load_state_dict(mapped, strict=False)
    print(f"checkpoint: {ckp_path}")
    print(f"  {len(mapped)} keys remapped, {len(missing)} missing, {len(unexpected)} unexpected")
    if unmapped:
        print(f"  unmapped legacy keys ({len(unmapped)}): {unmapped[:4]}")
    if len(missing) > 10:
        print(f"  note: {len(missing)} missing keys, first few: {list(missing)[:4]}")
    net.eval()

    actor = net.actor_net
    mask = mean_mask()
    has_mem = actor.timeline_layer is not None

    def logits(data) -> np.ndarray:
        with torch.inference_mode():
            return actor(data).squeeze(0).numpy()

    def logits_no_goal(data) -> np.ndarray:
        """Goal slot zeroed (same weights, no goal information)."""
        original = actor.encoder.goal_slot
        dim = actor.cfg.feature_dim
        actor.encoder.goal_slot = lambda canonical_x, d: canonical_x.new_zeros(1, dim)
        try:
            return logits(data)
        finally:
            actor.encoder.goal_slot = original

    def logits_no_mem(data) -> np.ndarray:
        """Timeline removed, so the memory conditioning is not built at all."""
        timeline, actor.timeline_layer = actor.timeline_layer, None
        try:
            return logits(data)
        finally:
            actor.timeline_layer = timeline

    swap_same = 0
    d_goal: list[float] = []
    d_ablate: list[float] = []
    d_mem: list[float] = []
    spread: list[float] = []
    agree_oracle: list[float] = []
    overlap: list[float] = []
    n_pairs = 0

    # FiLM strength: is the generator reading its conditioning input at all, or
    # has it collapsed onto its own bias (i.e. a learned constant)?
    goal_film = actor.films.generators["goal"]
    bias_norm = float(goal_film.generator.bias.detach().norm())
    h_norm: list[float] = []
    wh_norm: list[float] = []
    d_gamma: list[float] = []
    gamma_norm: list[float] = []
    d_beta: list[float] = []
    beta_norm: list[float] = []
    prev: tuple[np.ndarray, np.ndarray] | None = None

    def modulation(data) -> tuple[np.ndarray, np.ndarray]:
        with torch.inference_mode():
            canonical_x = actor.encoder.encode("canonical", data)
            h = actor.encoder.goal_slot(canonical_x, data)
            gamma, beta = goal_film.params(h)
            wh = goal_film.generator.weight.detach() @ h.squeeze(0)
        h_norm.append(float(h.norm()))
        wh_norm.append(float(wh.norm()))
        return gamma.squeeze(0).numpy(), beta.squeeze(0).numpy()

    for _ in range(args.episodes):
        (x0, _), (y0, _) = scene.sample_task()
        (_, _), (y1, _) = scene.sample_task()
        # SAMPLE-variant rows are redrawn on every update_nodes(), so both
        # exports are taken from the same RNG state to isolate the goal.
        seed = int(np.random.randint(1 << 30))

        np.random.seed(seed)
        graph.set_goal(y0)
        graph.set_start(x0)  # set_start() refreshes the canonical + goal rows
        data_a = graph.export()
        keys_a = list(graph._export_keys)
        la = logits(data_a)

        np.random.seed(seed)
        graph.set_goal(y1)  # swap the goal, keep the same start
        graph.set_start(x0)  # the flow training uses: refresh nodes + edges
        data_b = graph.export()
        keys_b = list(graph._export_keys)
        lb = logits(data_b)

        map_a = dict(zip(keys_a, la.tolist()))
        map_b = dict(zip(keys_b, lb.tolist()))
        shared = [k for k in keys_a if k in map_b]
        if not shared:
            continue

        n_pairs += 1
        va = np.array([map_a[k] for k in shared])
        vb = np.array([map_b[k] for k in shared])
        spread.append(float(va.std()))
        swap_same += int(max(map_a, key=map_a.get) == max(map_b, key=map_b.get))
        d_goal.append(float(np.abs(va - vb).mean()))
        overlap.append(len(shared) / len(set(keys_a) | set(keys_b)))

        abl = dict(zip(keys_a, logits_no_goal(data_a).tolist()))
        d_ablate.append(float(np.mean([abs(map_a[k] - abl[k]) for k in shared])))
        if has_mem:
            mem = dict(zip(keys_a, logits_no_mem(data_a).tolist()))
            d_mem.append(float(np.mean([abs(map_a[k] - mem[k]) for k in shared])))

        gamma, beta = modulation(data_a)
        gamma_norm.append(float(np.linalg.norm(gamma)))
        beta_norm.append(float(np.linalg.norm(beta)))
        if prev is not None:
            d_gamma.append(float(np.linalg.norm(gamma - prev[0])))
            d_beta.append(float(np.linalg.norm(beta - prev[1])))
        prev = (gamma, beta)

        scores = effect_scores(data_a, mask)
        best = keys_a[int(np.argmax(scores))]
        agree_oracle.append(float(max(map_a, key=map_a.get) == best))

    def stat(xs: list[float]) -> str:
        return f"{np.mean(xs):.3f} (median {np.median(xs):.3f})" if xs else "-"

    print(f"\n{args.episodes} starts, each with two goals ({n_pairs} usable pairs)")
    print(f"  logit spread across options      : {stat(spread)}")
    print(f"  |dlogit| goal swap               : {stat(d_goal)}")
    print(f"  |dlogit| goal slot zeroed        : {stat(d_ablate)}")
    if has_mem:
        print(f"  |dlogit| memory removed          : {stat(d_mem)}")
    print(f"  options shared by both goals     : {stat(overlap)}")
    print(f"\n  argmax invariant under goal swap : {swap_same}/{n_pairs} "
          f"({100.0 * swap_same / max(n_pairs, 1):.1f}%)")
    print(f"  argmax == best effect match      : {np.mean(agree_oracle) * 100:.1f}% "
          f"({int(np.sum(agree_oracle))}/{n_pairs})")

    def m(xs: list[float]) -> float:
        return float(np.mean(xs)) if xs else 0.0

    print("\n  actor FiLM / goal generator")
    print(f"    |h_goal|                          : {m(h_norm):.3f}")
    print(f"    |W h_goal| / |bias|               : {m(wh_norm) / max(bias_norm, 1e-9):.4f}"
          f"   (|W h| {m(wh_norm):.3f}, |bias| {bias_norm:.3f})")
    print(f"    |gamma|                           : {m(gamma_norm):.3f}")
    print(f"    |gamma(A)-gamma(B)| / |gamma|     : {m(d_gamma) / max(m(gamma_norm), 1e-9):.4f}"
          "   (goal-driven share of the modulation)")
    print(f"    |beta(A)-beta(B)| / |beta|        : {m(d_beta) / max(m(beta_norm), 1e-9):.4f}")
    print("\n  A high argmax invariance together with a small |dlogit|/spread would mean the "
          "goal\n  barely enters the decision. Note the greedy effect-match rule is only a "
          "one-step\n  heuristic: for multi-step goals the right first option is often a "
          "precondition enabler.")


if __name__ == "__main__":
    main()
