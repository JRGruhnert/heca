import sys
import types

import torch

from heca.misc import logger

# NOTE: device selection follows TAPAS
# (https://github.com/robot-learning-freiburg/TAPAS.git), but asks torch instead of
# parsing nvidia-smi: several ranks start at once, and a shared temp file makes
# that a race.


def get_gpu_with_most_free_mem() -> int:
    """Index of the GPU with the most free memory."""
    free = [
        torch.cuda.mem_get_info(index)[0] for index in range(torch.cuda.device_count())
    ]
    return int(max(range(len(free)), key=free.__getitem__))


def use_device(index: int) -> torch.device:
    """Pin this process to GPU ``index``."""
    global gpu_no, device
    if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
        return device

    index = int(index) % torch.cuda.device_count()
    if device.type == "cuda" and device.index == index:
        return device  # already where the import-time pick put us

    gpu_no = index
    device = torch.device(f"cuda:{index}")
    torch.cuda.set_device(device)
    logger.info(f"Running on {device} (pinned by rank)")
    return device


use_gpu = torch.cuda.is_available()
gpu_no = get_gpu_with_most_free_mem() if use_gpu else None
device = torch.device("cuda:{}".format(gpu_no) if use_gpu else "cpu")
if use_gpu:
    torch.cuda.set_device(device)
logger.info(f"Running on {device}")


_TAPAS_SELECT_GPU = "tapas_gmm.utils.select_gpu"


def _forward_to_hardware(attribute: str):
    """Resolve a name on the shim module from this one, lazily."""
    if attribute.startswith("__"):
        raise AttributeError(attribute)
    try:
        return globals()[attribute]
    except KeyError as exc:
        raise AttributeError(
            f"{_TAPAS_SELECT_GPU}.{attribute} is not provided by heca's device "
            f"shim (heca.misc.hardware has no attribute {attribute!r})"
        ) from exc


def _shim_tapas_gpu_selection() -> None:
    existing = sys.modules.get(_TAPAS_SELECT_GPU)
    if existing is not None:
        logger.warning(
            f"{_TAPAS_SELECT_GPU} was imported before heca: it picked its own "
            f"device ({getattr(existing, 'device', '?')}) instead of {device}"
        )
        return

    shim = types.ModuleType(_TAPAS_SELECT_GPU)
    shim.__doc__ = (
        "heca shim: answers tapas_gmm's device selection from heca.misc.hardware "
        "instead of running nvidia-smi through a shared temp file."
    )
    shim.__all__ = ("use_gpu", "gpu_no", "device", "get_gpu_with_most_free_mem")  # type: ignore
    shim.__getattr__ = _forward_to_hardware  # type: ignore
    sys.modules[_TAPAS_SELECT_GPU] = shim


_shim_tapas_gpu_selection()
