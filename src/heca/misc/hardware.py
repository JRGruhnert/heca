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


def use_device(index: int) -> torch.device:
    """Pin this process to GPU ``index`` (``index`` is taken modulo the GPU count).

    The import-time selection below asks "which GPU is freest right now", which
    is evaluated independently in every process: ranks started together race onto
    the same card and the split changes from run to run. The distributed spawner
    therefore calls this with the rank, so each rank gets a fixed, balanced GPU.
    Returns the device this process now uses.
    """
    global gpu_no, device
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return device

    index = int(index) % torch.cuda.device_count()
    if device.type == "cuda" and device.index == index:
        return device  # already where the import-time pick put us

    gpu_no = index
    device = torch.device(f"cuda:{index}")
    # pin it, so bare "cuda" and anything else asking for a device agree with the
    # choice made here
    torch.cuda.set_device(device)
    logger.info(f"Running on {device} (pinned by rank)")
    return device


use_gpu = torch.cuda.is_available()
gpu_no = get_gpu_with_most_free_mem() if use_gpu else None
device = torch.device("cuda:{}".format(gpu_no) if use_gpu else "cpu")
if use_gpu:
    torch.cuda.set_device(device)
logger.info(f"Running on {device}")
