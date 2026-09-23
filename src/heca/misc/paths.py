import os
from pathlib import Path

ENV_FILE_VAR = "HECA_ENV_FILE"
DATA_VAR = "HECA_DATA"
OUT_VAR = "HECA_OUT"
ENV_FILE_NAME = ".env"


def find_repo_root(start: Path | None = None) -> Path:
    """The nearest ancestor of ``start`` that contains ``pyproject.toml``."""
    start = Path(__file__).resolve() if start is None else Path(start).resolve()
    for path in (start, *start.parents):
        if (path / "pyproject.toml").exists():
            return path
    raise RuntimeError("Could not find repository root")


def load_env_file(path: Path | None = None) -> Path | None:
    """Read ``KEY=VALUE`` lines from an env file into ``os.environ``.

    Blank lines and ``#`` comments are skipped, surrounding quotes are stripped
    and ``$VAR``/``${VAR}`` references are expanded. Existing environment
    variables are never overwritten. Returns the file that was read, or ``None``.
    """
    if path is None:
        env = os.environ.get(ENV_FILE_VAR)
        path = Path(env).expanduser() if env else find_repo_root() / ENV_FILE_NAME
    if not path.exists():
        return None
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key.strip(), os.path.expandvars(value))
    return path


def data_root() -> Path:
    """Root of all persisted data: ``$HECA_DATA`` or ``<repository>/data``."""
    env = os.environ.get(DATA_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return find_repo_root() / "data"


def output_root() -> Path:
    """Root of run outputs (plots, logs): ``$HECA_OUT`` or the repository."""
    env = os.environ.get(OUT_VAR)
    if env:
        return Path(env).expanduser().resolve()
    return find_repo_root()


# read the env file before anything asks for a root
ENV_FILE_LOADED = load_env_file()
