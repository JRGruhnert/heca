"""Log the condition-scoring values for the real scene models.

For every agent (default: all scenes/models, filter with --scene/--model) this
prints, per entity:

  - change score + anchor/TARGET classification (ConPair.change_scores vs
    ANCHOR_CHANGE_THRESHOLD),
  - score_single validity of the demo pre/post values under their own models
    (the [0,1] pose+state gate score, averaged over all demos),
  - pre<->post containment (how much of the post distribution sits inside the
    pre distribution; ~1 for anchors, low for entities that move),
  - update_nodes behavior on a goal-conditioned graph built from this agent's
    conditions: whether each post-node adopts the goal, the start (anchor), or
    a fresh sample.

Value-logging counterpart of the old assertion suite; see
a08_view_anchor_entities.py for the same style.
"""

import argparse
import json
import sys
from pathlib import Path

# Make ``conf`` and ``scripts.common`` importable when run directly.
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

import numpy as np

from heca.data.data import DCEntity, DCScene
from heca.experts.expert import ExpertModel
from heca.graphs.graph import ANCHOR_CHANGE_THRESHOLD, Graph
from heca.graphs.node import OptionNode
from heca.misc import logger

from scripts.common.args import add_model_argument, add_scene_argument
from scripts.common.scenes import agents_by_scene


def _params(model):
    return model.get_parameters().copy()


def entity_stats(pair, label: str) -> dict:
    """Score every demo row under its own pre/post models + containment.

    Mean of ``score_single`` over *all* demo rows of the entity (the whole
    ``data_raw``, not a subset): how typical the recorded start/end values are
    under the fitted pre/post StepMix models.
    """
    ent = pair.pre.entities[label]
    pre_rows = np.asarray(pair.pre.data_raw[label])
    post_rows = np.asarray(pair.post.data_raw[label])
    pre_scores = [
        ent.score_single(r, _params(pair.pre.models[label]))[0] for r in pre_rows
    ]
    post_scores = [
        ent.score_single(r, _params(pair.post.models[label]))[0] for r in post_rows
    ]
    pre_post_cont = ent.containment_score(
        _params(pair.pre.models[label]), _params(pair.post.models[label])
    )
    return {
        "n_demos": int(pair.pre.data_raw[label].shape[0]),
        "pre_score": float(np.mean(pre_scores)),
        "post_score": float(np.mean(post_scores)),
        "pre_post_containment": float(pre_post_cont),
    }


def update_nodes_report(pair, anchors: set[str]) -> dict[str, dict]:
    """Build a goal-conditioned graph from this agent's conditions and report,
    per post-node, which source ``update_nodes`` used (goal / start / sample).

    Mirrors Graph.generate's node construction for one agent (see the old
    a07 G3 test) but on the real conditions.
    """
    label = pair.label
    entities = pair.pre.entities
    graph = Graph(entities=entities)
    pre_comp = graph.set_comps(label, pair.pre)
    post_comp = graph.set_comps(label, pair.post)
    pre_src = graph.set_precon(label, pair.pre, pre_comp)
    post_src = graph.set_postcon(
        label, pair.post, post_comp, pre_src, change_scores=pair.change_scores
    )
    graph.ns_option.add(
        "opt_" + label, OptionNode(model=None, sources={src for src in post_src.values()})
    )
    graph.es_stepmix.edges_from_sets(graph.ns_entity, graph.ns_entity)
    graph.es_summary.edges_from_sets(graph.ns_entity, graph.ns_option)
    graph.es_tapas.edges_from_sets(graph.ns_entity, graph.ns_entity)

    def make_scene(rows: dict[str, np.ndarray]) -> DCScene:
        ents = {}
        for k, v in rows.items():
            v = np.asarray(v, dtype=np.float32)
            ents[k] = DCEntity(value=v, feature=entities[k].gnn_format(v))
        return DCScene(ents)

    # Demo row 0 as start (pre values) and goal (post values).
    start = make_scene({k: pair.pre.data_raw[k][0] for k in entities})
    goal = make_scene({k: pair.post.data_raw[k][0] for k in entities})
    graph.set_goal(goal)
    graph.set_start(start)  # triggers update_nodes

    report: dict[str, dict] = {}
    for key in graph.goal_keys:
        node = graph.ns_entity.get_by_key(key)
        entity = node.entity
        if np.allclose(node.data.value, start.get(entity).value):
            source = "start (anchor)" if entity in anchors else "start"
        elif np.allclose(node.data.value, goal.get(entity).value):
            source = "goal"
        else:
            source = "sampled"
        report[entity] = {"key": key, "source": source, "anchor": entity in anchors}
    return report


def agent_report(cfg: ExpertModel.Config, reload: bool = False) -> dict:
    """Compute the scoring values of one agent's conditions."""
    logger.info(f"[{cfg.scene.tag}] Loading agent {cfg.tag}")
    agent = ExpertModel.get(cfg, auto_load=False)
    if reload:
        agent.force_recompute()

    pair = agent.conditions
    scores = pair.change_scores
    anchors = {e for e, s in scores.items() if s < ANCHOR_CHANGE_THRESHOLD}

    entities: dict[str, dict] = {}
    for entity, score in sorted(scores.items()):
        entities[entity] = {
            "change_score": float(score),
            "anchor": entity in anchors,
            **entity_stats(pair, entity),
        }

    return {
        "scene": cfg.scene.tag,
        "tag": cfg.tag,
        "entities": entities,
        "update_nodes": update_nodes_report(pair, anchors),
    }


def print_report(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(
        "Condition scoring (anchor threshold = "
        f"{ANCHOR_CHANGE_THRESHOLD} normalized units)"
    )
    print("=" * 72)

    total_entities = 0
    total_anchors = 0
    for rec in records:
        print(f"\n[{rec['scene']}] agent: {rec['tag']}")
        if not rec["entities"]:
            print("    (no shared entities between pre and post)")
            continue
        n_anchors = 0
        for entity, info in rec["entities"].items():
            total_entities += 1
            kind = "ANCHOR" if info["anchor"] else "TARGET"
            n_anchors += int(info["anchor"])
            print(
                f"    {entity:<12} change={info['change_score']:7.3f} "
                f"[{kind:<7}] n={info['n_demos']:4d} "
                f"pre={info['pre_score']:.3f} post={info['post_score']:.3f} "
                f"pre_post_cont={info['pre_post_containment']:.3f}"
            )
        total_anchors += n_anchors
        print("    update_nodes (goal-conditioned post nodes):")
        for entity, info in sorted(rec["update_nodes"].items()):
            tag = " (anchor)" if info["anchor"] else ""
            print(f"      {entity:<12} -> {info['source']}{tag}")
        print(f"    -> {n_anchors} anchors, {len(rec['entities']) - n_anchors} targets")

    print("\n" + "-" * 72)
    print(
        f"Totals: {len(records)} agents, {total_entities} entity-scores, "
        f"{total_anchors} anchors"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    parser.add_argument(
        "--use-gt",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use ground-truth observations for the conditions (default: true). "
        "Disable to extract values from images (loads the image encoders).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Refit the conditions from demos instead of loading conditions.joblib.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to write the results as JSON.",
    )
    args = parser.parse_args()

    records = []
    failures = []
    for scene_cfg, model_cfgs in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        for cfg in model_cfgs:
            if args.model and cfg.tag != args.model:
                continue
            try:
                agent = ExpertModel.get(cfg, auto_load=False)
                if not args.use_gt:
                    agent.use_gt(False)
                records.append(agent_report(cfg, reload=args.reload))
            except Exception as exc:  # keep going if one agent fails
                logger.error(f"[{scene_cfg.tag}] agent {cfg.tag} failed: {exc}")
                failures.append(
                    {"scene": scene_cfg.tag, "tag": cfg.tag, "error": str(exc)}
                )

    print_report(records)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(
                {
                    "anchor_threshold": ANCHOR_CHANGE_THRESHOLD,
                    "agents": records,
                    "failures": failures,
                },
                indent=2,
            )
        )
        logger.info(f"Wrote JSON report to {out_path}")

    if failures:
        logger.warning(f"{len(failures)} agent(s) failed; see JSON report.")
        sys.exit(1)


if __name__ == "__main__":
    main()
