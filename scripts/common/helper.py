from pathlib import Path


def fmt_duration(seconds: float) -> str:
    total = int(round(seconds))
    if total < 1:
        return "<1s"
    hours, rest = divmod(total, 3600)
    minutes, secs = divmod(rest, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def find_checkpoint(run_dir: Path, name: str) -> Path:
    """A named checkpoint in a run dir, or the highest-numbered ``ckp_*.pt``."""
    if name != "latest":
        return run_dir / name
    checkpoints = sorted(
        run_dir.glob("ckp_*.pt"), key=lambda p: int(p.stem.split("_")[1])
    )
    if not checkpoints:
        raise FileNotFoundError(f"no ckp_*.pt in {run_dir}")
    return checkpoints[-1]
