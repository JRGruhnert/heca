import argparse
import json
import sys
from pathlib import Path

from heca.data.entity import Entity

# Make ``conf`` and ``scripts.common`` importable when run directly as
# ``python scripts/report_change_scores.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

from heca.experts.expert import ExpertModel
from heca.misc import logger

from scripts.common.args import (
    add_model_argument,
    add_reload_argument,
    add_scene_argument,
    add_tag_argument,
    add_use_gt_argument,
)
from scripts.common.scenes import agents_by_scene


def agent_report(cfg: ExpertModel.Config, reload: bool = False) -> dict:
    """Return the per-entity change scores of one agent's conditions."""
    logger.info(f"[{cfg.scene.tag}] Loading agent {cfg.tag}")
    agent = ExpertModel.get(cfg, auto_load=False)
    if reload:
        agent.force_recompute()

    pair = agent.conditions
    scores = pair.change_scores

    entities: dict[str, dict] = {}
    for entity, score in sorted(scores.items()):
        n_samples = None
        try:
            n_samples = int(pair.pre.data_raw[entity].shape[0])
        except (KeyError, AttributeError, IndexError):
            pass
        entities[entity] = {
            "change_score": float(score),
            "anchor": bool(score < Entity.ANCHOR_THRESHOLD),
            "n_samples": n_samples,
        }
    return {
        "scene": cfg.scene.tag,
        "tag": cfg.tag,
        "entities": entities,
    }


def print_report(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Pre->post change scores (anchor threshold = {Entity.ANCHOR_THRESHOLD} std)")
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
            n = info["n_samples"] if info["n_samples"] is not None else "?"
            print(
                f"    {entity:<24} change={info['change_score']:>8.3f} std  "
                f"[{kind:<7}] (n={n})"
            )
        total_anchors += n_anchors
        n_targets = len(rec["entities"]) - n_anchors
        print(f"    -> {n_anchors} anchors, {n_targets} targets")

    print("\n" + "-" * 72)
    print(
        f"Totals: {len(records)} agents, {total_entities} entity-scores, "
        f"{total_anchors} anchors"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    add_reload_argument(parser)
    args = parser.parse_args()

    records = []
    for scene_cfg, model_cfgs in agents_by_scene():
        if args.scene and scene_cfg.tag != args.scene:
            continue
        for cfg in model_cfgs:
            if args.model and cfg.tag != args.model:
                continue
            agent = ExpertModel.get(cfg, auto_load=False)
            agent.use_gt(args.gt)
            records.append(agent_report(cfg, reload=args.reload))

    print_report(records)


if __name__ == "__main__":
    main()
