"""Track the visual keypoint read over the TAPAS trajectories instead of fitting.

Called exactly like ``scripts/b01_fit_experts.py`` -- same scene / encoder /
model / gt / threads / jobs arguments, one worker per scene -- but where the fit
would build a TPGMM it only records the image encoder's error against ground
truth, timestep by timestep.

Two things are taken from the fitting code rather than reinvented:

* the demos are the model config's preselection (``segment_ids``), i.e. the
  trajectories the fit is set up on;
* the read runs the way ``TapasExpert.load_demos`` reads them: states and extras
  off (``use_states = use_extra = False`` -- the TPGMM never sees them) and the
  particle filter **not** reset between the frames of one trajectory. The reset
  happens once per demo, so each read is conditioned on the belief carried from
  the frames before it. That is what makes this different from
  ``d07_error_plots``, which resets per frame and so measures independent frames.

Per (expert model, keypoint backend) one CSV is written to ``<repo>/visual``::

    drawer0_a_b_dino_errors.csv
    drawer0_a_b_roma_errors.csv

columns: ``demo,timestep,entity,gt_x,gt_y,gt_z,vis_x,vis_y,vis_z,error,
error_offset,score``

``error`` is ``|vis - gt|`` in metres; ``error_offset`` additionally applies the
per-entity reference offset ``d07_error_plots`` uses (the read in the frame the
reference was clicked in, against the ground truth there), which removes a
systematic keypoint bias; ``score`` is the encoder's confidence for that read.

    python -m scripts.b06_visual_trajectory_errors --scene scene0 \
        --model drawer0_a_b --kp-backend dino --max-demos 3

A model preselected 20 demos of ~40-60 frames each, so bound a first run with
``--model`` / ``--max-demos`` / ``--max-frames``.
"""

import argparse
import csv
import multiprocessing as mp
import signal
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial
from pathlib import Path

import h5py
import numpy as np

# make ``conf`` and ``scripts.common`` importable when run as a file, not as -m
_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from heca.data.reference import Reference
from heca.experts.expert import ExpertModel
from heca.experts.tapas import TapasExpert
from heca.learning import dist as pdist
from heca.misc import logger
from heca.misc.paths import find_repo_root
from heca.scenes.scene import Scene

from scripts.common.args import (
    add_encoder_arguments,
    add_model_argument,
    add_scene_argument,
    add_threads_argument,
    add_use_gt_argument,
)
from scripts.common.helper import fmt_duration
from scripts.common.scenes import (
    experts_for_scene,
    scene_config,
    scene_tags,
    select_encoder,
)

COLUMNS = [
    "demo",
    "timestep",
    "segment",
    "segment_t",
    "entity",
    "gt_x",
    "gt_y",
    "gt_z",
    "vis_x",
    "vis_y",
    "vis_z",
    "error",
    "error_offset",
    "error_segment",
    "score",
]


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def scene_dataset(scene: Scene, labels: list[str]) -> Path:
    """The scene recording that holds the frames the references were clicked in."""
    root = Path(scene.save_dir(scene.cfg))
    for path in sorted(root.glob("*.h5")):
        with h5py.File(path, "r") as probe:
            if "rgb" in probe and f"heca_{labels[0]}_pos" in probe:
                return path
    raise SystemExit(f"no dataset with frames and ground truth in {root}")


def reference_offsets(
    scene: Scene, labels: list[str], dataset: Path, kp_extractor
) -> dict[str, np.ndarray]:
    """Per-entity offset between the keypoint read and the truth at the reference.

    The same quantity ``d07_error_plots`` subtracts: the read in the frame the
    reference was clicked in, against the ground truth of that same frame.
    """
    from scripts.d07_error_plots import frame_at

    offsets: dict[str, np.ndarray] = {}
    with h5py.File(dataset, "r") as file:
        for label in labels:
            reference = scene.references[label][Reference.POSITION]
            frame = frame_at(file, reference.frame, scene)
            kp_extractor.reset_tracking()
            keypoint = kp_extractor.extract_poses(frame)[label]
            truth = np.asarray(file[f"heca_{label}_pos"][reference.frame], dtype=float)
            offsets[label] = truth - keypoint.xyz
            logger.info(
                f"{label}: reference offset {np.round(offsets[label], 3)} m "
                f"(read {np.round(keypoint.xyz, 3)})"
            )
    return offsets


def segment_ids(expert: TapasExpert, scene_frames: list) -> np.ndarray:
    """Per-frame segment index, with TAPAS's own velocity segmentation.

    The fit builds one TPGMM segment per velocity pause and expresses each
    segment in the frame at its start; ``_velocity_segments`` on the packed
    action reproduces exactly the boundaries ``load_demos`` uses.
    """
    was = expert._use_gt
    expert._use_gt = True  # the conversion is GT-only, as in _scan_demos
    try:
        td = expert.dcscenes_to_tdtapas(scene_frames)
    finally:
        expert._use_gt = was
    vel = np.linalg.norm(td.action[:, :3].numpy().astype(float), axis=1)
    seg_cfg = expert.cfg.policy.model.demos_segmentation
    boundaries = expert._velocity_segments(vel, seg_cfg)
    frames = np.arange(len(scene_frames))
    return np.searchsorted(np.asarray(boundaries), frames, side="right")


def preselected_demos(expert: TapasExpert) -> list[int]:
    """The demo ids the model config preselects (``segment_ids``)."""
    selected = expert.cfg.segment_ids
    if selected:
        logger.info(f"{expert.cfg.tag}: {len(selected)} preselected demo(s)")
        return list(selected)
    ids = expert._all_demo_ids()
    logger.info(
        f"{expert.cfg.tag}: no segment_ids, using all {len(ids)} demo(s) of the file"
    )
    return ids


def track_tapas(
    expert: TapasExpert,
    scene: Scene,
    out_dir: Path,
    backend: str,
    demo_limit: int,
    demos: list[int] | None,
    max_frames: int,
    offsets_on: bool,
) -> int:
    """The fitting pass with the fit removed: read every demo, record the error."""
    started = time.perf_counter()
    # exactly what load_demos does: the TPGMM takes pos + rot only
    expert.use_states = False
    expert.use_extra = False
    expert.use_gt(False)  # builds the keypoint encoder, renders the frames

    labels = list(scene.entities)
    demo_ids = demos or preselected_demos(expert)
    if demo_limit:
        demo_ids = demo_ids[:demo_limit]
    if not demo_ids:
        logger.warning(f"{expert.cfg.tag}: no demo selected, skipped")
        return 0

    offsets: dict[str, np.ndarray] = {}
    if offsets_on:
        try:
            offsets = reference_offsets(
                scene, labels, scene_dataset(scene, labels), expert.kp_extractor
            )
        except (SystemExit, KeyError) as exc:
            logger.warning(f"{expert.cfg.tag}: no reference offsets ({exc})")

    rows: list[list] = []
    demos_path = TapasExpert.load_dir(expert.cfg) / "demos.h5"
    with h5py.File(demos_path, "r") as file:
        for demo_id in demo_ids:
            scenes, images = scene.load_dataset(
                file, selections=[demo_id], with_images=True
            )
            scene_frames, image_frames = scenes[0], images[0]
            seg = segment_ids(expert, scene_frames)  # DCScenes, before truncation
            if max_frames:
                scene_frames = scene_frames[:max_frames]
                image_frames = image_frames[:max_frames]
                seg = seg[:max_frames]
            expert.reset_tracking()  # one belief chain per trajectory, as in the fit

            truth: list[dict[str, np.ndarray]] = []
            seen: list[dict[str, tuple[np.ndarray, float]]] = []
            for step, image in enumerate(image_frames):
                poses = expert.kp_extractor.extract_poses(image)
                frame_truth: dict[str, np.ndarray] = {}
                frame_seen: dict[str, tuple[np.ndarray, float]] = {}
                for label in labels:
                    raw = np.asarray(scene_frames[step][label].value[:3], dtype=float)
                    keypoint = poses[label]
                    frame_truth[label] = scene.unnormalize_position(raw)
                    frame_seen[label] = (
                        np.asarray(keypoint.xyz, dtype=float),
                        float(keypoint.score),
                    )
                truth.append(frame_truth)
                seen.append(frame_seen)

            # per segment: align at the segment's first frame, the way the fit
            # expresses every segment in the frame at its start
            segment_offset: dict[tuple[int, str], np.ndarray] = {}
            for segment in sorted(set(int(v) for v in seg)):
                first = int(np.flatnonzero(seg == segment)[0])
                for label in labels:
                    segment_offset[(segment, label)] = (
                        truth[first][label] - seen[first][label][0]
                    )

            # 0 -> 1 within each segment, the x axis of the per-segment plots
            segment_t = np.zeros(len(image_frames))
            for segment_value in set(int(v) for v in seg):
                where = np.flatnonzero(seg == segment_value)
                if len(where) > 1:
                    segment_t[where] = np.arange(len(where)) / (len(where) - 1)

            for step in range(len(image_frames)):
                segment = int(seg[step])
                for label in labels:
                    point, score = seen[step][label]
                    error = float(np.linalg.norm(point - truth[step][label]))
                    corrected = float(
                        np.linalg.norm(
                            point + offsets.get(label, np.zeros(3)) - truth[step][label]
                        )
                    )
                    per_segment = float(
                        np.linalg.norm(
                            point
                            + segment_offset.get((segment, label), np.zeros(3))
                            - truth[step][label]
                        )
                    )
                    rows.append(
                        [
                            demo_id,
                            step,
                            segment,
                            f"{float(segment_t[step]):.4f}",
                            label,
                            *[f"{value:.6f}" for value in truth[step][label]],
                            *[f"{value:.6f}" for value in point],
                            f"{error:.6f}",
                            f"{corrected:.6f}",
                            f"{per_segment:.6f}",
                            f"{score:.4f}",
                        ]
                    )
            logger.info(
                f"  demo {demo_id}: {len(image_frames)} frame(s) read, "
                f"{len(set(int(v) for v in seg))} segment(s)"
            )

    if not rows:
        logger.warning(f"{expert.cfg.tag} / {backend}: nothing measured")
        return 0

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{expert.cfg.tag}_{backend}_errors.csv"
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(COLUMNS)
        writer.writerows(rows)

    by_entity: dict[str, list[tuple[float, float, float]]] = {}
    for row in rows:
        by_entity.setdefault(row[4], []).append(
            (float(row[11]), float(row[12]), float(row[13]))
        )
    for label in sorted(by_entity):
        values = np.asarray([item[0] for item in by_entity[label]])
        corrected = np.asarray([item[1] for item in by_entity[label]])
        per_segment = np.asarray([item[2] for item in by_entity[label]])
        logger.info(
            f"[{expert.cfg.tag} {backend}] {label:10s} n={len(values):5d} "
            f"error mean {values.mean():.3f} / median {np.median(values):.3f} / "
            f"max {values.max():.3f} m | reference-offset {corrected.mean():.3f} m "
            f"| segment-offset {per_segment.mean():.3f} m"
        )
    logger.info(
        f"wrote {path} ({len(rows)} row(s), "
        f"{fmt_duration(time.perf_counter() - started)})"
    )
    return 1


# --------------------------------------------------------------------------- #
# pipeline, invoked the way b01_fit_experts invokes its own
# --------------------------------------------------------------------------- #
def pipeline_model(
    scene_cfg: Scene.Config,
    cfg: ExpertModel.Config,
    backend: str,
    out_dir: Path,
    demo_limit: int,
    demos: list[int] | None,
    max_frames: int,
    offsets_on: bool,
) -> None:
    """Track one model -- the analogue of b01's pipeline_scene, one per worker.

    One job per model rather than per scene, because a model is ~1-2k encoded
    frames and they are independent: this is what makes a four-model run finish
    in an afternoon instead of a week.
    """
    out_dir = Path(out_dir)
    scene = Scene.get(scene_cfg)
    logger.info(f"[{scene_cfg.tag}] === trajectory errors for {cfg.tag} ({backend}) ===")
    expert = ExpertModel.get(cfg, auto_load=False)
    assert isinstance(expert, TapasExpert), "only TAPAS experts have demo files"
    written = track_tapas(
        expert,
        scene,
        out_dir,
        backend,
        demo_limit,
        demos,
        max_frames,
        offsets_on,
    )
    logger.info(f"[{scene_cfg.tag} / {cfg.tag} / {backend}] {written} csv file(s) written")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    add_scene_argument(parser)
    add_encoder_arguments(parser)
    add_model_argument(parser)
    add_use_gt_argument(parser)
    add_threads_argument(parser)
    for action in parser._actions:
        if action.dest == "kp_backend":
            action.choices = ("dino", "roma", "all")
            action.default = "all"
    parser.add_argument(
        "--max-demos",
        type=int,
        default=3,
        help="Demos per model, after the config's preselection (default 3, "
        "0 = every preselected demo).",
    )
    parser.add_argument(
        "--demos",
        nargs="+",
        type=int,
        default=None,
        help="Explicit demo ids instead of the config's segment_ids.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Truncate every trajectory to its first N frames (0 = all).",
    )
    parser.add_argument(
        "--no-offsets",
        action="store_true",
        help="Skip the reference offsets (error_offset then equals error).",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output directory (default: <repo>/visual).",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=0,
        help="Max models tracked concurrently, one encoder per worker (0 = one "
        "process per model/backend job, 1 = sequential). Each worker needs a few "
        "GB of RAM.",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt))

    out_dir = Path(args.out) if args.out else find_repo_root() / "visual"
    backends = ["dino", "roma"] if args.kp_backend == "all" else [args.kp_backend]
    if args.gt:
        logger.warning(
            "--gt belongs to the fitting pipeline: the read here is always the "
            "image encoder's, and the ground truth is only the comparison target"
        )

    for backend in backends:
        jobs = []
        for tag in scene_tags():
            if args.scene and tag != args.scene:
                continue
            # the selection mutates the shared configs, so the jobs of one
            # backend must be submitted before the next backend selects
            select_encoder(tag, args.backbone, args.stride, backend)
            models = [
                model
                for model in experts_for_scene(tag)
                if not (args.model and model.tag != args.model)
            ]
            for model in models:
                jobs.append((scene_config(tag), model))
        if not jobs:
            continue
        workers = len(jobs) if args.jobs <= 0 else min(args.jobs, len(jobs))
        logger.info(
            f"[{backend}] tracking {len(jobs)} model(s) with {workers} process(es) "
            f"(each worker loads its own encoder)."
        )
        threads = pdist.threads_for(workers, args.threads)
        pool = ProcessPoolExecutor(
            max_workers=workers,
            initializer=partial(pdist.pin_intra_op_threads, threads),
            mp_context=mp.get_context("spawn"),
        )
        futures = [
            pool.submit(
                pipeline_model,
                scene_cfg,
                model,
                backend,
                out_dir,
                args.max_demos,
                args.demos,
                args.max_frames,
                not args.no_offsets,
            )
            for scene_cfg, model in jobs
        ]
        try:
            for future in as_completed(futures):
                future.result()
        finally:
            # the workers share this process group, so an interrupt reaches them
            pool.shutdown(wait=False, cancel_futures=True)


if __name__ == "__main__":
    main()
