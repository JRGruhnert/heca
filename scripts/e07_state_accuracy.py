"""State readout accuracy: three seeds x 250 images, aggregated over all states.

Only the correct answer counts -- the per-state split is not reported. The
encoder answers each frame on its own, so a frame contributes one hit or miss per
state-bearing entity. Writes the per-seed rates and their mean +- std to JSON; no
plot is produced.

    python -m scripts.e07_state_accuracy --model clip
    python -m scripts.e07_state_accuracy --model molmo --samples 250 --seeds 0 1 2
"""

import argparse
import json
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from heca.image_encoders.clip_encoder import ClipEncoder
from heca.image_encoders.image_encoder import ImageEncoder
from heca.image_encoders.molmo_encoder import MolmoEncoder
from heca.image_encoders.tipsy_encoder import TipsyEncoder
from heca.misc import logger
from heca.scenes.scene import Scene

from scripts.common.frames import dataset, frame_at, rows, truth_state
from scripts.common.scenes import scene_config

BUILDERS = {
    "clip": ClipEncoder.Config,
    "tipsy": TipsyEncoder.Config,
    "molmo": MolmoEncoder.Config,
}


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--model", choices=BUILDERS, default="clip")
    parser.add_argument("--samples", type=int, default=250, help="images per seed")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument(
        "--out", default="", help="default: the scene's folder as state-accuracy-<model>.json"
    )
    args = parser.parse_args()

    scene = Scene.get(scene_config(args.scene))
    encoder = ImageEncoder.get(BUILDERS[args.model]())
    encoder.prepare_for_scene(scene.cfg)
    file_path = dataset(scene)
    out = (
        Path(args.out)
        if args.out
        else Path(scene.save_dir(scene.cfg)) / f"state-accuracy-{args.model}.json"
    )

    per_seed: list[float] = []
    classes: list[int] = []
    with h5py.File(file_path, "r") as file:
        with_labels = [l for l, e in scene.entities.items() if e.cfg.n_states > 1]
        truth = {
            label: np.asarray(file[f"heca_{label}_ste"]).ravel() for label in with_labels
        }
        classes = sorted({int(v) for t in truth.values() for v in np.unique(t)})
        per_class = max(args.samples // (len(with_labels) * len(classes)), 1)
        logger.info(
            f"{args.model}: {len(with_labels)} entities, classes {classes}, "
            f"{per_class} per (entity, class) per seed"
        )
        for seed in args.seeds:
            rng = np.random.default_rng(seed)
            cases: list[tuple[str, int]] = []
            for label in with_labels:
                for value in classes:
                    pool = np.flatnonzero(truth[label] == value)
                    picks = rng.choice(pool, size=min(per_class, len(pool)), replace=False)
                    cases += [(label, int(row)) for row in picks]
            rng.shuffle(cases)

            cache: dict[int, dict] = {}
            hits = seen = 0
            for done, (label, row) in enumerate(cases, start=1):
                if row not in cache:
                    cache[row] = encoder.extract_states(frame_at(file, row, scene))
                found = cache[row].get(label)
                if found is None:
                    continue
                seen += 1
                hits += int(found[0] == int(truth[label][row]))
                if done % 100 == 0 or done == len(cases):
                    logger.info(f"{args.model} seed {seed}: {done}/{len(cases)} cases")
            per_seed.append(hits / max(seen, 1))
            logger.info(
                f"{args.model} seed {seed}: {hits}/{seen} = {100 * hits / max(seen, 1):.1f}% "
                f"({len(cache)} frames, balanced over {len(classes)} classes x {len(with_labels)} entities)"
            )

    values = np.asarray(per_seed, dtype=float)
    payload = {
        "model": args.model,
        "scene": args.scene,
        "dataset": file_path.name,
        "samples_per_seed": args.samples,
        "seeds": list(args.seeds),
        "accuracy_per_seed": [float(value) for value in values],
        "mean": float(values.mean()),
        "std": float(values.std()),
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")
    logger.info(
        f"{args.model}: {100 * payload['mean']:.1f}% +- {100 * payload['std']:.1f} "
        f"(std over {len(values)} seeds) -> {out}"
    )


if __name__ == "__main__":
    main()
