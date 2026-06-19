import h5py
import numpy as np

from heca.agents.experts.expert import ExpertAgent

cfg = ExpertAgent.Config()

load_path = ExpertAgent.resolve(cfg) / "demos.h5"
save_path = ExpertAgent.resolve(cfg) / "demos_new.h5"

file = h5py.File(load_path, "r")

data = {}
for k in file.keys():
    data[k] = np.asarray(file[k])

bs = []

for value in data["priviliged_button0_state"]:
    bs.append(value)

data["priviliged_button0_state"] = np.array(bs)
with h5py.File(save_path, "w") as f:
    for key, value in data.items():
        f.create_dataset(
            key,
            data=value,
            compression="gzip",
        )
    f.close()
file.close()
