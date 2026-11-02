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
    with h5py.File(file_path, "r") as file:
        total = len(file["rgb"])
        for seed in args.seeds:
            hits = seen = 0
            for done, row in enumerate(rows(total, args.samples, seed), start=1):
                row = int(row)
                for label, (state, _) in encoder.extract_states(
                    frame_at(file, row, scene)
                ).items():
                    seen += 1
                    hits += int(state == truth_state(file, label, row))
                if done % 50 == 0 or done == args.samples:
                    logger.info(f"{args.model} seed {seed}: {done}/{args.samples} frames")
            per_seed.append(hits / max(seen, 1))
            logger.info(f"{args.model} seed {seed}: {hits}/{seen} = {100 * hits / max(seen, 1):.1f}%")

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
