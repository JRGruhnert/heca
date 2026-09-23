import os
import socket
from datetime import timedelta

import torch
import torch.distributed as dist

from heca.misc import hardware, logger

DEFAULT_PORT = 29500


def pin_intra_op_threads(threads: int = 1) -> None:
    """Cap BLAS/OpenMP threads.

    The per-step tensors here are tiny, so intra-op parallelism is overhead:
    measured 0.42 s vs 1.91 s for the same call at 1 vs 8 threads. One thread per
    process, many processes, is the configuration that scales.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ.setdefault(var, str(threads))
    torch.set_num_threads(threads)


def init(rank: int, world_size: int, port: int = DEFAULT_PORT) -> None:
    """Join the process group (gloo: everything here runs on CPU).

    The timeout bounds every collective: a peer that dies mid-training aborts the
    others instead of leaving them waiting forever for a barrier that never comes.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=5),
    )


def active() -> bool:
    return dist.is_available() and dist.is_initialized()


def rank() -> int:
    return dist.get_rank() if active() else 0


def world_size() -> int:
    return dist.get_world_size() if active() else 1


def is_main() -> bool:
    """True on rank 0 (or in single-process mode), i.e. the writer of artifacts."""
    return rank() == 0


def broadcast_tensors(tensors: dict[str, torch.Tensor], src: int = 0) -> None:
    """Send rank ``src``'s values to everyone; a lone process is already in sync."""
    if not active():
        return
    for t in tensors.values():
        dist.broadcast(t, src=src)


def mean_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """All-reduce to the mean, in place (FedAvg over one tensor per rank)."""
    if not active():
        return tensors
    n = world_size()
    for t in tensors.values():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(n)
    return tensors


def shutdown() -> None:
    if not active():
        return
    try:
        dist.barrier()
    except Exception:
        pass
    dist.destroy_process_group()


def _free_port() -> int:
    """An unused local port, so repeated spawns in one sweep never collide."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _entry(
    rank: int, worker, world_size: int, args: tuple, port: int, n_gpus: int
) -> None:
    # Give every rank a fixed GPU. Without this, each process independently picks
    # "the freest GPU" at import time (heca.misc.hardware), so ranks started
    # together race onto the same card and the split differs between runs.
    if n_gpus > 1:
        hardware.use_device(rank % n_gpus)
    init(rank, world_size, port)
    try:
        worker(rank, world_size, *args)
    finally:
        shutdown()


def spawn(worker, world_size: int, args: tuple = ()) -> None:
    """Run ``worker(rank, world_size, *args)`` in ``world_size`` processes.

    ``world_size == 1`` runs the worker in the current process, so the same code
    path covers single- and multi-client training.
    """
    if world_size <= 1:
        worker(0, 1, *args)
        return
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        per_gpu = [rank % n_gpus for rank in range(world_size)]
        counts = {gpu: per_gpu.count(gpu) for gpu in sorted(set(per_gpu))}
        split = ", ".join(f"cuda:{gpu}x{count}" for gpu, count in counts.items())
        logger.info(f"Assigning {world_size} ranks over {n_gpus} GPUs: {split}")
    torch.multiprocessing.spawn(
        _entry,
        nprocs=world_size,
        args=(worker, world_size, args, _free_port(), n_gpus),
        join=True,
    )
