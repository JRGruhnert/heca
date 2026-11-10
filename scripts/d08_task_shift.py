"""Distribution shift of entity start positions: task setups vs randomized mode.

Samples 200 episode starts from the randomized training distribution (once,
shared by every task) and 200 starts per task setup (task mode). For each task
one figure is written: per entity and positional axis (x, y, z), the fitted
Gaussians of the task setup (solid) and the randomized distribution (dashed).
Positions are metres, position block only.

    python -m scripts.d08_task_shift --scene scene0 --samples 200

The samples are cached next to the figures (JSON), so replots are instant;
pass --refresh to re-sample.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless plotting

import matplotlib.pyplot as plt
import numpy as np
from heca.misc import logger
from heca.misc.paths import output_root
from heca.scenes.scene import Scene
from scripts.common.scenes import scene_config

AXES = ["x", "y", "z"]


def sample_positions(
    scene: Scene, samples: int, seed_from: int = 0
) -> dict[str, list[list[float]]]:
    """entity -> list of [x, y, z] metres for ``samples`` episode starts."""
    positions: dict[str, list] = {}
    for i in range(samples):
        scene.seed = seed_from + i
        (x, _), (y, _) = scene.sample_task(with_scenes=True)
        for label in scene.entities:
            raw = np.asarray(x[label].value[:3], dtype=float)
            metres = scene.unnormalize_position(raw)
            positions.setdefault(label, []).append([float(v) for v in metres])
        if (i + 1) % 50 == 0 or i + 1 == samples:
            logger.info(f"  {i + 1}/{samples} starts sampled")
    return positions


def pca_basis(samples: np.ndarray) -> np.ndarray:
    """PCA components of a pooled cloud, sorted by variance (D, D)."""
    centred = samples - samples.mean(axis=0)
    cov = centred.T @ centred / max(len(centred), 1)
    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    return vectors[:, order]


def gaussian_ellipse(mean, cov, n_std: float, **kwargs):
    """One iso-probability ellipse of a 2D Gaussian at ``n_std`` standard deviations."""
    from matplotlib.patches import Ellipse

    values, vectors = np.linalg.eigh(cov)
    order = np.argsort(values)[::-1]
    values, vectors = values[order], vectors[:, order]
    angle = float(np.degrees(np.arctan2(vectors[1, 0], vectors[0, 0])))
    width, height = 2.0 * n_std * np.sqrt(np.maximum(values, 0.0))
    return Ellipse(mean, width, height, angle=angle, fill=False, **kwargs)


def mahalanobis_shift(a: np.ndarray, r: np.ndarray) -> float:
    """Distance between the two means, measured in randomized-covariance units."""
    cov = np.cov(r, rowvar=False)
    diff = a.mean(axis=0) - r.mean(axis=0)
    return float(np.sqrt(diff @ np.linalg.pinv(cov) @ diff))


def entity_frames(
    random_positions: dict[str, list], tasks: dict[str, dict], entities: list[str]
) -> dict[str, tuple]:
    """Per entity: PCA basis of the randomized cloud + a box covering every cloud.

    The basis and the box are the same across all task figures, so the panels
    are directly comparable, and a degenerate cloud (line or point) still gets
    a proper square box instead of collapsing.
    """
    frames: dict[str, tuple] = {}
    for label in entities:
        rand = np.asarray(random_positions[label], dtype=float)
        centre = rand.mean(axis=0)
        components = pca_basis(rand)[:, :2]
        span = float(np.abs((rand - centre) @ components).max())
        for task in tasks.values():
            task_samples = np.asarray(task["positions"][label], dtype=float)
            span = max(span, float(np.abs((task_samples - centre) @ components).max()))
        frames[label] = (components, centre, span)
    return frames


def plot_pca_task(
    task_id: int,
    task_positions: dict[str, list],
    random_positions: dict[str, list],
    entities: list[str],
    plot_entities: list[str],
    frames: dict[str, tuple],
    out: Path,
    stem: str,
) -> dict[str, float]:
    """One panel per entity in ``plot_entities``; the shifts cover all ``entities``."""
    shifts = {
        label: mahalanobis_shift(
            np.asarray(task_positions[label], dtype=float),
            np.asarray(random_positions[label], dtype=float),
        )
        for label in entities
    }
    fig, axes = plt.subplots(
        1, len(plot_entities), figsize=(1.7 * len(plot_entities), 1.7), squeeze=False
    )
    for c, label in enumerate(plot_entities):
        ax = axes[0, c]
        components, centre, span = frames[label]
        task_samples = np.asarray(task_positions[label], dtype=float)
        rand_samples = np.asarray(random_positions[label], dtype=float)
        a2 = (task_samples - centre) @ components
        r2 = (rand_samples - centre) @ components

        ax.scatter(
            a2[:, 0], a2[:, 1], s=6, color="mediumspringgreen", alpha=0.4, linewidths=0
        )
        ax.scatter(r2[:, 0], r2[:, 1], s=4, color="hotpink", alpha=0.4, linewidths=0)
        half = 1.15 * max(span, 1e-4)
        ax.set_xlim(-half, half)
        ax.set_ylim(-half, half)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout(pad=0.0, w_pad=0.5, h_pad=0.0)
    path = out / f"task_shift_pca_{stem}_task{task_id}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"wrote {path}")
    return shifts


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out",
        type=Path,
        default=output_root() / "plots",
        help="directory for the figures and the sample cache",
    )
    parser.add_argument(
        "--refresh", action="store_true", help="re-sample instead of the cache"
    )
    parser.add_argument(
        "--exclude",
        nargs="+",
        default=["button0", "button1"],
        help="Entities left out of the figures (default: button0 button1 -- they "
        "sit at one fixed point, so their panel carries no cloud). The "
        "Mahalanobis shifts are still reported for them.",
    )
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"task_shift_{args.scene}.json"

    cached = None
    if cache.exists() and not args.refresh:
        try:
            cached = json.loads(cache.read_text())
            if cached.get("samples") != args.samples:
                cached = None
        except (json.JSONDecodeError, OSError):
            cached = None
    if cached:
        logger.info(f"{cache.name}: reusing the cached samples")
    else:
        scene = Scene.get(scene_config(args.scene))
        scene.set_mode("randomized")
        scene.task_id = None
        logger.info(f"randomized: sampling {args.samples} starts")
        random_positions = sample_positions(scene, args.samples, args.seed)

        tasks: dict[str, dict] = {}
        scene.set_mode("task")
        for task_id in range(1, len(scene.tasks) + 1):
            scene.task_id = task_id
            name = str(scene.tasks[task_id - 1].get("task_name", task_id))
            logger.info(f"task {task_id} ({name}): sampling {args.samples} starts")
            tasks[str(task_id)] = {
                "task_name": name,
                "positions": sample_positions(scene, args.samples, args.seed),
            }
        payload = {
            "scene": args.scene,
            "samples": args.samples,
            "seed": args.seed,
            "randomized": random_positions,
            "tasks": tasks,
        }
        cache.write_text(json.dumps(payload, indent=2) + "\n")
        logger.info(f"cached the samples to {cache}")
        cached = payload

    random_positions = cached["randomized"]
    entities = sorted(random_positions)
    plot_entities = [label for label in entities if label not in set(args.exclude)]
    if not plot_entities:
        raise SystemExit(f"--exclude removed every entity of {args.scene}")
    frames = entity_frames(random_positions, cached["tasks"], plot_entities)
    metrics: dict[str, dict] = {}
    for task_id, task in cached["tasks"].items():
        shifts = plot_pca_task(
            int(task_id),
            task["positions"],
            random_positions,
            entities,
            plot_entities,
            frames,
            out_dir,
            args.scene,
        )
        metrics[str(task_id)] = {"task_name": task.get("task_name"), **shifts}
        logger.info(
            f"task {task_id}: Mahalanobis shift "
            + ", ".join(f"{label}={value:.2f}" for label, value in shifts.items())
        )
    metrics_path = out_dir / f"task_shift_pca_{args.scene}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    logger.info(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()
