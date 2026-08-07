from collections import defaultdict
from dataclasses import dataclass
import numpy as np
import h5py

from heca.agents.experts.expert import ExpertAgent
from heca.scenes.scene import Scene
from heca.misc.base import Configurable


class DemoAutoparser(Configurable):
    @dataclass(kw_only=True)
    class Config(Configurable.Config):
        agent: ExpertAgent.Config
        scene: Scene.Config
        dataset_name: str = "visual-scene-play-v0.h5"
        file_name: str = "demos.h5"
        random_ep: bool = True

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        with h5py.File(self.dataset_path, "r") as f:
            done = np.asarray(f["oracle_done"])[:, 0]
            success = np.asarray(f["oracle_success"])[:, 0]
            task = np.asarray(f["privileged_target_task"], dtype=str)

            # Find segment boundaries (indices where oracle switches)
            done_idxs = np.where(done == 1.0)[0]
            seg_starts = [0] + list(done_idxs)
            seg_ends = list(done_idxs) + [len(done)]

            # Bucket successful segments by task + direction
            buckets = defaultdict(list)  # key → list of (start, end) index tuples

            for start, end in zip(seg_starts, seg_ends):
                if start >= end:
                    continue
                # Success is judged at the last step BEFORE the switch
                last_step = end - 1
                if success[last_step] != 1.0:
                    continue

                bucket_key = self._bucket_key(f, task[last_step], start, last_step)
                buckets[bucket_key].append((start, end - 1))

            # Write each bucket to a separate HDF5 file
            all_keys = list(f.keys())
            for bucket_key, segments in buckets.items():
                out_path = self.output_dir / f"{bucket_key}.h5"
                with h5py.File(out_path, "w") as out:
                    demo_id = 0
                    for start, end in segments:
                        demo_ids = np.full(end - start + 1, demo_id, dtype=np.int32)
                        for key in all_keys:
                            data = f[key][start : end + 1]
                            if "demo" not in out and key == "demo":
                                continue  # skip if demo dataset doesn't exist yet
                            if key not in out:
                                maxshape = (None,) + data.shape[1:]
                                out.create_dataset(
                                    key,
                                    data=data,
                                    maxshape=maxshape,
                                    chunks=(1,) + data.shape[1:],
                                )
                            else:
                                ds = out[key]
                                n = ds.shape[0]
                                ds.resize(n + len(data), axis=0)
                                ds[n : n + len(data)] = data
                        # Add demo_id dataset
                        if "demo" not in out:
                            out.create_dataset(
                                "demo", data=demo_ids[:1], maxshape=(None,), chunks=(1,)
                            )
                        else:
                            ds = out["demo"]
                            n = ds.shape[0]
                            ds.resize(n + len(demo_ids), axis=0)
                            ds[n : n + len(demo_ids)] = demo_ids
                        demo_id += 1

                print(f"  {bucket_key}: {len(segments)} demos → {out_path}")

    def _bucket_key(self, f, task_name: str, seg_start: int, seg_end: int) -> str:
        """Return a bucket key like 'cube_pick' or 'faucet_open'."""
        base = task_name.rsplit("_", 1)[
            0
        ]  # "button_0" → "button", "faucet_0" → "faucet"

        # Determine direction for joint-based objects
        if base in ("faucet", "doorlock", "lever", "drawer", "window"):
            pos_key = f"privileged_{task_name}_pos"
            target_key = f"heca_target_{task_name}_pos"
            start_val = f[pos_key][seg_start][0]
            target_val = f[target_key][seg_end][0]
            direction = "open" if target_val > start_val else "close"
            return f"{base}_{direction}"

        # Button: pressed or unpressed
        if base == "button":
            start_state = f[f"privileged_{task_name}_state"][seg_start][0]
            direction = "press" if start_state == 0 else "unpress"
            return f"{base}_{direction}"

        # Cube / peg / lid: pick-and-place (always both in one oracle)
        if base in ("cube", "peg", "lid"):
            return f"{base}_pick_place"

        return base
