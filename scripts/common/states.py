"""Sample one state readout and score it against the recorded ground truth.

The readout answers a per-entity question; the recording stores the answer the
environment actually had, on the same frame, so no reference offset is involved.
"""

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import h5py
from heca.experts.expert import ExpertModel
from heca.image_encoders.image_encoder import ImageEncoder
from heca.misc import logger
from heca.scenes.scene import Scene

from scripts.common.frames import dataset, frame_at, rows, truth_state
from scripts.common.scenes import experts_for_scene, scene_config

ENCODERS: dict[str, Callable[[], ImageEncoder.Config]] = {}


def register(name: str, build: Callable[[], ImageEncoder.Config]):
    ENCODERS[name] = build


def parse(model: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"Sample the {model} state readout.")
    parser.add_argument("--scene", default="scene0")
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default="", help="default: the scene's folder")
    return parser.parse_args()


def sample(model: str) -> Path:
    args = parse(model)
    scene = Scene.get(scene_config(args.scene))
    encoder = ImageEncoder.get(ENCODERS[model]())
    encoder.prepare_for_scene(scene.cfg)
    file_path = dataset(scene)
    out = (
        Path(args.out)
        if args.out
        else Path(scene.save_dir(scene.cfg)) / f"states-{model}.json"
    )
    records = []
    with h5py.File(file_path, "r") as file:
        picked = rows(len(file["rgb"]), args.samples, args.seed)
        for done, row in enumerate(picked, start=1):
            states = encoder.extract_states(frame_at(file, int(row), scene))
            for label, (state, probability) in states.items():
                records.append(
                    {
                        "label": label,
                        "row": int(row),
                        "truth": truth_state(file, label, int(row)),
                        "predicted": int(state),
                        "probability": float(probability),
                    }
                )
            if done % 10 == 0 or done == len(picked):
                logger.info(f"{done}/{len(picked)} frames read with {model}")
    out.write_text(
        json.dumps(
            {"model": model, "scene": args.scene, "samples": args.samples,
             "seed": args.seed, "dataset": file_path.name, "records": records},
            indent=1,
        )
        + "\n"
    )
    logger.info(f"wrote {out}")
    return out
