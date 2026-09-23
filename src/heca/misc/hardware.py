import sys
import types

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


# --- tapas_gmm's own GPU picker must never run ------------------------------
#
# ``tapas_gmm.utils.select_gpu`` picks a GPU at import time by shelling out to
# nvidia-smi through a *shared* temp file (``/tmp/gpu_mem``) which it deletes
# again. Ranks that import it together clobber that file, and whichever rank
# opens it after another rank's ``rm`` dies with
# ``FileNotFoundError: '/tmp/gpu_mem'`` (a half-written file gives
# ``IndexError``/``ValueError`` instead). heca selects the device itself — and
# under the spawner pins one per rank — so answer that module from here.

_TAPAS_SELECT_GPU = "tapas_gmm.utils.select_gpu"


def _forward_to_hardware(attribute: str):
    """Resolve a name on the shim module from this one, lazily.

    Reading through to ``hardware`` (instead of copying values) means a rank
    that is re-pinned after this point still sees the pinned device. Dunders are
    never forwarded: ``__file__``/``__path__``/``__spec__`` must stay absent so
    the module keeps looking like a plain module to the import system.
    """
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
    shim.__all__ = ("use_gpu", "gpu_no", "device", "get_gpu_with_most_free_mem")
    shim.__getattr__ = _forward_to_hardware  # PEP 562
    sys.modules[_TAPAS_SELECT_GPU] = shim


_shim_tapas_gpu_selection()
