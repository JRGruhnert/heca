"""Frames and ground truth of a scene's recorded episodes, shared by the readouts."""

import h5py
import numpy as np
from pathlib import Path

from heca.data.data import TDImage
from heca.scenes.scene import Scene

IMAGE_KEYS = {"rgb", "depth", "mask", "extrinsics", "intrinsics", "demo"}


def dataset(scene: Scene) -> Path:
    """The recorded file that carries frames and per-entity ground truth."""
    root = Path(scene.save_dir(scene.cfg))
    for path in sorted(root.glob("*.h5")):
        with h5py.File(path, "r") as probe:
            if "rgb" in probe and any(key.startswith("heca_") for key in probe):
                return path
    raise SystemExit(f"no dataset with frames and ground truth in {root}")


def frame_at(file: h5py.File, row: int, scene: Scene) -> TDImage:
    image = {key: file[key][row] for key in IMAGE_KEYS if key != "demo"}
    extra = {key: file[key][row] for key in file.keys() if key not in IMAGE_KEYS}
    obs, _ = scene.to_internal(image, extra)
    return scene.to_td_image(obs)


def truth_pos(file: h5py.File, label: str, row: int) -> np.ndarray:
    return np.asarray(file[f"heca_{label}_pos"][row], dtype=float)


def truth_state(file: h5py.File, label: str, row: int) -> int:
    return int(np.ravel(file[f"heca_{label}_ste"][row])[0])


def rows(total: int, samples: int, seed: int = 0) -> np.ndarray:
    """The frames to read: spread over the recording, not the first N."""
    if samples >= total:
        return np.arange(total)
    return np.sort(np.random.default_rng(seed).choice(total, samples, replace=False))
