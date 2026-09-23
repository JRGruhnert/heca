import torch

from heca.misc import logger

# NOTE: device selection follows TAPAS
# (https://github.com/robot-learning-freiburg/TAPAS.git), but asks torch instead of
# parsing nvidia-smi: several ranks start at once, and a shared temp file makes
# that a race. TODO: Want to maybe add ROCm support


def get_gpu_with_most_free_mem() -> int:
    """Index of the GPU with the most free memory."""
    free = [
        torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
    ]
    return int(max(range(len(free)), key=free.__getitem__))


use_gpu = torch.cuda.is_available()
gpu_no = get_gpu_with_most_free_mem() if use_gpu else None
device = torch.device("cuda:{}".format(gpu_no) if use_gpu else "cpu")
if use_gpu:
    # pin it, so bare "cuda" and anything else asking for a device agree with the
    # choice made here
    torch.cuda.set_device(device)
logger.info(f"Running on {device}")
