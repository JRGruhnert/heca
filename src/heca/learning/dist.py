import os
import socket
import tempfile
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
from torch.multiprocessing.spawn import start_processes
from heca.misc import hardware, logger

DEFAULT_PORT = 29500
DEFAULT_TIMEOUT_MIN = 30.0


def collective_timeout() -> timedelta:
    """Timeout for one collective on the process group.

    A rank that is merely slow (oversubscribed node, GPU contention) must not
    take down an otherwise healthy shard, so the default is generous.
    Override with HECA_PG_TIMEOUT_MINUTES=<minutes> (e.g. 5 to fail fast).
    """
    raw = os.environ.get("HECA_PG_TIMEOUT_MINUTES", "").strip()
    if not raw:
        return timedelta(minutes=DEFAULT_TIMEOUT_MIN)
    try:
        minutes = float(raw)
    except ValueError:
        minutes = 0.0
    if minutes <= 0:
        logger.warning(
            f"HECA_PG_TIMEOUT_MINUTES={raw!r} is not a positive number, "
            f"using {DEFAULT_TIMEOUT_MIN:g} min"
        )
        return timedelta(minutes=DEFAULT_TIMEOUT_MIN)
    return timedelta(minutes=minutes)


def threads_for(ranks: int, threads: int = 0) -> int:
    """How many BLAS/OpenMP threads one rank may use."""
    if threads > 0:
        return threads
    cores = (
        len(os.sched_getaffinity(0))
        if hasattr(os, "sched_getaffinity")
        else os.cpu_count()
    )
    return max(1, int(cores or 1) // max(1, ranks))


def pin_intra_op_threads(threads: int = 1) -> None:
    """Cap BLAS/OpenMP/torch threads of this process to threads."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = str(threads)
    torch.set_num_threads(threads)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        return
    threadpool_limits(threads)  # applies now, not on __enter__


def init(rank: int, world_size: int, port: int = DEFAULT_PORT) -> None:
    """Join the process group (gloo: everything here runs on CPU).

    The collective timeout comes from HECA_PG_TIMEOUT_MINUTES (default 30):
    too low and a slow-but-alive rank kills the shard.
    """
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(port))
    timeout = collective_timeout()
    dist.init_process_group(
        backend="gloo",
        rank=rank,
        world_size=world_size,
        timeout=timeout,
    )
    if rank == 0:
        logger.info(
            f"gloo ready: world_size={world_size}, collective timeout "
            f"{timeout.total_seconds() / 60:g} min (HECA_PG_TIMEOUT_MINUTES)"
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


def gather_objects(values: Any) -> list[Any]:
    if not active():
        return [values]
    out: list[Any] = [None] * world_size()
    dist.all_gather_object(out, values)
    return out


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


def use_pyg_home(rank: int) -> None:
    """Point this process at its own PyG template cache.

    NOTE: A Workaround for the following problem:
        PyG renders jinja templates for ``propagate``/``edge_update`` on first use into
        a *per-user* directory (``$PYG_HOME``, default ``~/.cache/pyg/<tmp_dirname>/``)
        and imports the rendered file straight after writing it. Every rank of a launch
        writes the same path, so a rank can exec a file that another rank has just
        truncated (or a half-written one) and dies with
        One directory per rank, under node-local tmp, removes the sharing entirely and
        keeps the writes off a networked home directory. ``get_home_dir()`` reads the
        environment on every call, so setting it here (before any layer is built) is
        enough.
    """
    os.environ["PYG_HOME"] = os.path.join(
        tempfile.gettempdir(), "heca_pyg", f"rank{rank}"
    )


def _entry(
    rank: int,
    worker,
    world_size: int,
    args: tuple,
    port: int,
    n_gpus: int,
    threads: int,
) -> None:
    pin_intra_op_threads(threads)
    if n_gpus > 1:
        hardware.use_device(rank % n_gpus)
    use_pyg_home(rank)
    init(rank, world_size, port)
    try:
        worker(rank, world_size, *args)
    finally:
        shutdown()


def spawn(worker, world_size: int, args: tuple = (), threads: int = 0) -> None:
    threads = threads_for(world_size, threads)
    logger.info(f"Threads per rank: {threads} (world size {world_size})")
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus > 1:
        per_gpu = [rank % n_gpus for rank in range(world_size)]
        counts = {gpu: per_gpu.count(gpu) for gpu in sorted(set(per_gpu))}
        split = ", ".join(f"cuda:{gpu}x{count}" for gpu, count in counts.items())
        logger.info(f"Assigning {world_size} ranks over {n_gpus} GPUs: {split}")
    start_processes(
        _entry,
        nprocs=world_size,
        args=(worker, world_size, args, _free_port(), n_gpus, threads),
        join=True,
    )
