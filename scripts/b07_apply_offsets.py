"""Fill in the offset-corrected error column of b06 CSV files.

b06 writes ``visual/<expert>_<backend>_errors.csv``. When the machine running it
lacks the scene recording (``vi-scene0.h5``) the reference offsets cannot be
computed and the ``error_offset`` column equals ``error``. The raw ``gt_*`` and
``vis_*`` columns are kept, so the correction can be applied later on a machine
that has the recording:

    python -m scripts.b07_apply_offsets --csv-dir visual --scene scene0

For every ``<expert>_<backend>_errors.csv`` it recomputes the per-entity offsets
exactly like b06 (keypoint read at the reference frame vs the recording's
ground truth) and rewrites ``error_offset = ||vis + offset - gt||``.
Existing files are copied to ``<name>.pre_offset.csv`` before being overwritten.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import numpy as np  # noqa: E402

from heca.experts.expert import ExpertModel  # noqa: E402
from heca.misc import logger  # noqa: E402
from heca.scenes.scene import Scene  # noqa: E402
from scripts.b06_visual_trajectory_errors import (  # noqa: E402
    reference_offsets,
    scene_dataset,
)
from scripts.common.args import add_encoder_arguments  # noqa: E402
from scripts.common.scenes import (  # noqa: E402
    experts_for_scene,
    scene_config,
    select_encoder,
)

BACKENDS = ("dino", "roma")


def split_name(path: Path) -> tuple[str, str] | None:
    """``drawer0_a_b_dino_errors.csv`` -> (``drawer0_a_b``, ``dino``)."""
    base = path.stem.removesuffix("_errors")
    for backend in BACKENDS:
        suffix = f"_{backend}"
        if base.endswith(suffix):
            return base[: -len(suffix)], backend
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    add_encoder_arguments(ap)
    ap.add_argument("--scene", default="scene0")
    ap.add_argument("--model", default=None, help="only this expert tag")
    ap.add_argument("--csv-dir", type=Path, default=Path("visual"))
    for action in ap._actions:
        if action.dest == "kp_backend":
            action.choices = (*BACKENDS, "all")
            action.default = "all"
    args = ap.parse_args()

    paths = sorted(args.csv_dir.glob("*_errors.csv"))
    if not paths:
        raise SystemExit(f"no *_errors.csv in {args.csv_dir}")

    for path in paths:
        parts = split_name(path)
        if parts is None:
            logger.warning(f"{path.name}: unknown backend, skipped")
            continue
        expert_tag, backend = parts
        if args.kp_backend != "all" and backend != args.kp_backend:
            continue
        if args.model and expert_tag != args.model:
            continue

        select_encoder(args.scene, args.backbone, args.stride, backend)
        scene = Scene.get(scene_config(args.scene))
        cfg = next(
            (m for m in experts_for_scene(args.scene) if m.tag == expert_tag), None
        )
        if cfg is None:
            logger.warning(f"{path.name}: no expert {expert_tag!r}, skipped")
            continue
        expert = ExpertModel.get(cfg, auto_load=False)
        # builds the keypoint encoder, exactly as b06's track_tapas does
        expert.use_gt(False)
        labels = list(scene.entities)
        try:
            offsets = reference_offsets(
                scene, labels, scene_dataset(scene, labels), expert.kp_extractor
            )
        except (SystemExit, KeyError) as exc:
            logger.warning(f"{path.name}: no reference offsets ({exc})")
            continue

        backup = path.with_name(f"{path.stem}.pre_offset.csv")
        if not backup.exists():
            shutil.copy2(path, backup)

        rows = []
        updated = 0
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            axis_fields = [f"error_offset_{axis}" for axis in ("x", "y", "z")]
            for axis_field in axis_fields:
                if axis_field not in fields:
                    fields.append(axis_field)
            for row in reader:
                offset = offsets.get(row["entity"], np.zeros(3))
                if offset is None:
                    offset = np.zeros(3)
                seen = np.array([row["vis_x"], row["vis_y"], row["vis_z"]], dtype=float)
                truth = np.array([row["gt_x"], row["gt_y"], row["gt_z"]], dtype=float)
                corrected = seen + offset - truth
                row["error_offset"] = f"{float(np.linalg.norm(corrected)):.6f}"
                for axis in ("x", "y", "z"):
                    row[f"error_offset_{axis}"] = f"{abs(float(corrected[('xyz'.index(axis))])):.6f}"
                updated += 1
                rows.append(row)
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        logger.info(
            f"{path.name}: {updated} row(s), offsets "
            + ", ".join(f"{k} {np.round(v, 3)}" for k, v in offsets.items())
        )


if __name__ == "__main__":
    main()
