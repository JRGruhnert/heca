"""Evaluation curves, drawn exactly like ``trainingsplot.py``.

Same figure as the training plots (same size, colours, EMA smoothing, mean+-std
band, legend and output naming), but the numbers are success rates over an
evaluation sweep instead of a training curve, so the parsing differs:

  * ``eval/<group>.csv``  -- written by ``scripts/c07_sweep.py``: one row per
    checkpoint, mean and std already averaged over the seeds of that group,
    header ``epoch,virt-container,virt-container-std,...``
  * ``eval/runs/<tag>.csv`` -- one row per checkpoint per run, no std; used as a
    fallback when no averaged file exists, and then averaged here over seeds the
    same way ``trainingsplot.py`` does.

A group therefore names either its averaged file or its per-seed run files:

    dict(label="Rel", kind="group", files=["scene0_final-a10-gt-virt"])
    dict(label="Rel", kind="runs",  files=["scene0_final-a10-gt-virt"])

Groups whose files are missing are skipped with a warning, so a plot can be made
while part of the sweep is still running.
"""

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

# Axis
X_COLUMN = "epoch"  # evaluation rows are keyed by checkpoint, not row number
DPI = 300  # 300 for papers, 150 for quick looks
SMOOTH_ALPHA = 0.1
XMIN = 0  # set to None to fit the data
YMIN = 0.0  # set to None to fit mean-std with padding
YMAX = 1.0  # set to None to fit mean+std with padding
PAD = 0.05  # fractional padding when a limit is None
COL_SEP = ","
# Figure
FIG_SIZE_CM = (9.0, 6.0)  # same default as the .sty
LINE_WIDTH = 0.5
BAND_ALPHA = 0.40  # 0.20 ~= xcolor "!20"
SHOW_BAND = True  # False -> mean lines only
LEGEND_LOC = "lower right"  # matplotlib "loc" keyword
GRID = False
COLORS = [
    "hotpink",
    "springgreen",
    "cyan",
    "steelblue",
    "darkmagenta",
    "darkorange",
    "red",
    "saddlebrown",
    "olive",
    "yellow",
]
FONT_SIZE = 8  # base / default text
LABEL_SIZE = 9  # axis labels ("Epoch", "Success rate")
TICK_SIZE = 7  # tick labels on both axes
LEGEND_SIZE = 7  # legend entries
TITLE_SIZE = 9  # axes title

# Where scripts/c07_sweep.py wrote its CSVs (its --out, default "eval").
EVAL_DIR = Path("eval")
OUT_DIR = Path("out")
# Protocols of the sweep, in the order they are reported.
PROTOCOLS = ("virt-container", "virt-real", "gt-real")


# ============================================================================
#  implementation (no need to touch below)
# ============================================================================
def group_file(name: str) -> Path:
    """The averaged file for one experiment group."""
    return EVAL_DIR / f"{name}.csv"


def run_files(name: str, seeds: int = 3) -> list[Path]:
    """The per-run files of one experiment group (seed suffix restored)."""
    return [EVAL_DIR / "runs" / f"{name}_s{i}.csv" for i in range(seeds)]


def has_rows(path: Path) -> bool:
    """A header-only CSV is a sweep that has not written any checkpoint yet."""
    try:
        with open(path, newline="") as fh:
            return len(list(csv.DictReader(fh, delimiter=COL_SEP))) > 0
    except FileNotFoundError:
        return False


def resolve_group(grp: dict, seeds: int = 3) -> tuple[str, list[Path]]:
    """``(kind, paths)`` for one group, preferring its averaged file.

    ``kind`` is ``"group"`` when the rows already carry a ``-std`` column, else
    ``"runs"`` and the files are aggregated over seeds here.
    """
    kind = grp.get("kind")
    names = grp["files"]
    if kind is None:
        kind = "group" if all(has_rows(group_file(n)) for n in names) else "runs"
    paths = (
        [group_file(n) for n in names]
        if kind == "group"
        else [p for n in names for p in run_files(n, grp.get("seeds", seeds))]
    )
    paths = [p for p in paths if has_rows(p)]
    # an averaged file can exist but still be empty early in a sweep: fall back
    # to the per-run files that do have rows
    if kind == "group" and not paths:
        return "runs", [p for n in names for p in run_files(n, grp.get("seeds", seeds))
                        if has_rows(p)]
    return kind, paths


def read_run(path, column, x_column, sep):
    """Return (x, y, std) for one CSV file; ``std`` is None without a column."""
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=sep))
    except FileNotFoundError:
        sys.exit(f"error: missing file {path!r}")
    if not rows:
        return np.empty(0), np.empty(0), None  # empty sweep file: no rows yet
    header = list(rows[0].keys())
    if column not in header:
        sys.exit(
            f"error: column {column!r} not in {path!r}\n"
            f"       available: {', '.join(header)}"
        )
    if x_column is not None and x_column not in header:
        sys.exit(f"error: column {x_column!r} not in {path!r}")

    def num(value):
        try:
            return float(value)
        except (TypeError, ValueError):
            return float("nan")

    y = np.array([num(r[column]) for r in rows], dtype=float)
    std_column = f"{column}-std"
    std = (
        np.array([num(r[std_column]) for r in rows], dtype=float)
        if std_column in header
        else None
    )
    if x_column is None:
        x = np.arange(1, len(y) + 1, dtype=float)
    else:
        x = np.array([num(r[x_column]) for r in rows], dtype=float)
    return x, y, std


def ema(y, alpha):
    """Exponentially weighted moving average.

    alpha in (0,1]: weight on the current sample. NaNs are carried forward
    (they only appear here where every seed is NaN at a given x, which is
    rare; carrying keeps the plotted line continuous).
    """
    if alpha is None or not (0 < alpha <= 1.0) or len(y) == 0:
        return y
    out = np.empty_like(y, dtype=float)
    prev = np.nan
    for i, v in enumerate(y):
        if np.isnan(v):
            out[i] = prev
            continue
        prev = v if np.isnan(prev) else alpha * v + (1.0 - alpha) * prev
        out[i] = prev
    return out


def aggregate(files, column, x_column, sep):
    """Mean and sample std across runs (row-aligned, NaN-tolerant)."""
    runs = [read_run(f, column, x_column, sep)[:2] for f in files]
    runs = [r for r in runs if len(r[0])]
    if not runs:
        return np.empty(0), np.empty(0), np.empty(0)
    x = runs[0][0]
    n = len(x)
    y = np.full((n, len(runs)), np.nan)
    for j, (_, yj) in enumerate(runs):
        m = min(n, len(yj))
        y[:m, j] = yj[:m]
    mean = np.nanmean(y, axis=1)
    if y.shape[1] > 1:
        with np.errstate(invalid="ignore"):
            std = np.nanstd(y, axis=1, ddof=1)  # sample std (n-1)
        std = np.nan_to_num(std, nan=0.0)  # rows with a single value
    else:
        std = np.zeros(n)  # one run -> no spread
    return x, mean, std


def series(kind, files, column, x_column, sep):
    """``(x, mean, std)`` of one group, averaged file or per-run files."""
    if kind == "group":
        # the averaged file already carries one row per checkpoint
        x, mean, std = read_run(files[0], column, x_column, sep)
        return x, mean, np.zeros_like(mean) if std is None else std
    return aggregate(files, column, x_column, sep)


def resolve_column(files, columns, sep):
    """Return the first name in `columns` that exists in the file header.

    Inspects the first file in `files`. Exits with a clear error if none of
    the candidates is present.
    """
    if isinstance(columns, str):
        columns = [columns]
    path = files[0]
    try:
        with open(path, newline="") as fh:
            header = csv.DictReader(fh, delimiter=sep).fieldnames or []
    except FileNotFoundError:
        sys.exit(f"error: missing file {path!r}")
    for col in columns:
        if col in header:
            return col
    sys.exit(
        f"error: none of {columns!r} found in {path!r}\n"
        f"       available: {', '.join(header)}"
    )


def plot(
    cap: int | None,
    title: str,
    axis_labels: bool,
    column: list[str],
    groups: list[dict],
    xlabel: str,
    ylabel: str,
):
    plt.rcParams.update(
        {
            "font.size": FONT_SIZE,
            "axes.labelsize": LABEL_SIZE,
            "axes.titlesize": TITLE_SIZE,
            "xtick.labelsize": TICK_SIZE,
            "ytick.labelsize": TICK_SIZE,
            "legend.fontsize": LEGEND_SIZE,
        }
    )
    fig_w = FIG_SIZE_CM[0] / 2.54
    fig_h = FIG_SIZE_CM[1] / 2.54
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    y_lo, y_hi, x_max = [], [], -np.inf
    drawn = 0
    for grp in groups:
        kind, files = resolve_group(grp)
        if not files:
            print(f"skipping {grp['label']!r}: no evaluation CSV for {grp['files']}")
            continue
        col = resolve_column(files, column, COL_SEP)
        x, mean, std = series(kind, files, col, X_COLUMN, COL_SEP)
        if len(x) == 0 or np.all(np.isnan(mean)):
            print(f"skipping {grp['label']!r}: no rows yet in {grp['files']}")
            continue
        mean = ema(mean, SMOOTH_ALPHA)
        std = ema(std, SMOOTH_ALPHA)
        x_max = max(x_max, float(np.nanmax(x)))
        color = COLORS[drawn % len(COLORS)]
        if SHOW_BAND and np.any(std > 1e-12):
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=color,
                alpha=BAND_ALPHA,
                linewidth=0,
            )
            y_lo.append(float(np.nanmin(mean - std)))
            y_hi.append(float(np.nanmax(mean + std)))
        else:
            y_lo.append(float(np.nanmin(mean)))
            y_hi.append(float(np.nanmax(mean)))
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            label=grp["label"],
        )
        drawn += 1

    if not drawn:
        plt.close(fig)
        print(f"nothing to plot for {title!r}: no group had data")
        return

    # ---- limits -----------------------------------------------------------
    lo, hi = min(y_lo), max(y_hi)
    span = hi - lo if hi - lo > 1e-12 else 1.0
    ax.set_xlim(XMIN, cap if cap is not None else x_max)
    ax.set_ylim(
        YMIN if YMIN is not None else lo - PAD * span,
        YMAX if YMAX is not None else hi + PAD * span,
    )

    if axis_labels:
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    ax.tick_params(direction="in", top=True, right=True)
    ax.grid(GRID)
    ax.legend(loc=LEGEND_LOC, frameon=False)
    fig.tight_layout()
    col_name = column if isinstance(column, str) else column[0]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = (
        OUT_DIR
        / f"eval_{title}_{cap}_{col_name}{'_labeled' if axis_labels else ''}.png"
    )
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
    print(f"wrote {output_path}")


def plot_grid(
    title: str,
    caps: list[int | None],
    columns: list[tuple[list[str], str]],
    groups: list[dict],
):
    """One figure per x-axis cap and column, labelled and unlabelled."""
    for cap in caps:
        for column in columns:
            plot(cap, title, True, column[0], groups, "Epoch", column[1])
            plot(cap, title, False, column[0], groups, "Epoch", column[1])


def group(label: str, name, **kwargs) -> dict:
    """One entry of a group list; ``name`` may be one tag or a list of tags."""
    names = [name] if isinstance(name, str) else list(name)
    return dict(label=label, files=names, **kwargs)


if __name__ == "__main__":
    # The sweep's checkpoints; None fits the x axis to the evaluated range.
    caps = [None]

    # Same groups as trainingsplot.py, resolved against the evaluation sweep.
    groupA = [
        group("Default", "scene0_final-a0-gt-virt"),
        group("Cond", "scene0_final-a2-gt-virt"),
        group("Sum", "scene0_final-a4-gt-virt"),
        group("Mem", "scene0_final-a6-gt-virt"),
        group("Rel", "scene0_final-a10-gt-virt"),
        group("Rel*", "scene0_final-a12-gt-virt"),
    ]
    columns = [(["virt-container"], "Success Rate")]
    plot_grid("standard", caps, columns, groupA)

    groupY = [
        group("Virtual", "scene0_final-a10-gt-virt"),
        group("Real", "scene0_final-a10-gt"),
        group("Virt (Fed)", "scene1_final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"),
        group("Real (Fed)", "scene1_final-a10-gt-fed-k1-mu0.1-alpha0.5"),
    ]
    columns = [(["virt-container", "gt-real"], "Success Rate")]
    plot_grid("comp", caps, columns, groupY)

    # The federated scene groups only appear once that sweep exists; the
    # plotter skips the missing ones with a warning.
    groupB = [
        group("Scene A", "scene1_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene B", "scene3_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene C", "scene5_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene D", "scene6_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene E", "scene7_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene F", "scene8_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene G", "scene9_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
        group("Scene H", "scene10_final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"),
    ]
    plot_grid("fed_a0", caps, columns, groupB)
