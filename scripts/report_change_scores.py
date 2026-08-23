import argparse
import json
import sys
from pathlib import Path

# Make ``conf`` and ``scripts.common`` importable when run directly as
# ``python scripts/report_change_scores.py`` (mirrors scripts/__init__.py).
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")  # headless (conditions fitting imports matplotlib)

from heca.experts.expert import ExpertModel
from heca.graphs.graph import ANCHOR_CHANGE_THRESHOLD
from heca.misc import logger

try:
    from scripts.common.args import (
        add_ee_argument,
        add_scene_argument,
        add_tag_argument,
    )
    from scripts.common.scenes import agents_by_scene, to_ee
except ModuleNotFoundError as exc:
    raise SystemExit(
        f"Could not import the scene/agent configs: {exc}\n"
        "Make sure the scene dependencies (e.g. the ogbench fork pinned in "
        "pyproject.toml) are installed in the active environment."
    ) from exc


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
            "anchor": bool(score < ANCHOR_CHANGE_THRESHOLD),
            "n_samples": n_samples,
        }
    return {
        "scene": cfg.scene.tag,
        "tag": cfg.tag,
        "entities": entities,
    }


def print_report(records: list[dict]) -> None:
    print("\n" + "=" * 72)
    print(f"Pre->post change scores (anchor threshold = {ANCHOR_CHANGE_THRESHOLD} std)")
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
            kind = "anchor" if info["anchor"] else "CHANGED"
            n_anchors += int(info["anchor"])
            n = info["n_samples"] if info["n_samples"] is not None else "?"
            print(
                f"    {entity:<24} change={info['change_score']:>8.3f} std  "
                f"[{kind:<7}] (n={n})"
            )
        total_anchors += n_anchors
        n_changed = len(rec["entities"]) - n_anchors
        print(f"    -> {n_anchors} anchor, {n_changed} changed")

    print("\n" + "-" * 72)
    print(
        f"Totals: {len(records)} agents, {total_entities} entity-scores, "
        f"{total_anchors} anchors"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_scene_argument(parser)
    add_tag_argument(parser)
    add_ee_argument(parser)
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
    for scene_tag, models in agents_by_scene().items():
        if args.scene and scene_tag != args.scene:
            continue
        for cfg in models:
            if args.tag and cfg.tag != args.tag:
                continue
            if args.ee:
                cfg = to_ee(cfg)
            try:
                agent = ExpertModel.get(cfg, auto_load=False)
                if not args.use_gt:
                    agent.use_gt(False)
                records.append(agent_report(cfg, reload=args.reload))
            except Exception as exc:  # keep going if one agent fails
                logger.error(f"[{scene_tag}] agent {cfg.tag} failed: {exc}")
                failures.append({"scene": scene_tag, "tag": cfg.tag, "error": str(exc)})

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
