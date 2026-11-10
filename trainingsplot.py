import csv
import sys

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np

# Axis
X_COLUMN = None  # None -> use the row index 1,2,3,...
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


# ============================================================================
#  implementation (no need to touch below)
# ============================================================================
def subfolder(folder: str) -> str:
    return "csv/federated" if "fed" in folder.lower() else "csv/standard"


def file_list(
    folder: list[str],
    scene: str = "scene0",
    seeds: int = 3,
) -> list[str]:
    return [
        f"{file_path(f, scene, subfolder(f))}_s{i}.csv"
        for i in range(seeds)
        for f in folder
    ]


def file_path(folder_name, scene_name, root_dir) -> str:
    return root_dir + "/" + folder_name + "/" + scene_name + "_" + folder_name


def read_run(path, column, x_column, sep):
    """Return (x, y) arrays for one CSV file."""
    try:
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter=sep))
    except FileNotFoundError:
        sys.exit(f"error: missing file {path!r}")
    if not rows:
        sys.exit(f"error: {path!r} is empty")
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
    if x_column is None:
        x = np.arange(1, len(y) + 1, dtype=float)
    else:
        x = np.array([num(r[x_column]) for r in rows], dtype=float)
    return x, y


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
    runs = [read_run(f, column, x_column, sep) for f in files]
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
    epoch: int,
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
    for idx, grp in enumerate(groups):
        col = resolve_column(grp["files"], column, COL_SEP)
        x, mean, std = aggregate(grp["files"], col, X_COLUMN, COL_SEP)
        mean = ema(mean, SMOOTH_ALPHA)
        std = ema(std, SMOOTH_ALPHA)
        x_max = max(x_max, float(np.nanmax(x)))
        if SHOW_BAND and np.any(std > 1e-12):
            ax.fill_between(
                x,
                mean - std,
                mean + std,
                color=COLORS[idx],
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
            color=COLORS[idx],
            linewidth=LINE_WIDTH,
            solid_capstyle="round",
            label=grp["label"],
        )

    # ---- limits -----------------------------------------------------------
    lo, hi = min(y_lo), max(y_hi)
    span = hi - lo if hi - lo > 1e-12 else 1.0
    ax.set_xlim(XMIN, epoch if epoch is not None else x_max)
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
    output_path = (
        f"out/{title}_{epoch}_{col_name}{'_labeled' if axis_labels else ''}.png"
    )
    fig.savefig(output_path, dpi=DPI)
    print(f"wrote {output_path}")


def plot_grid(
    title: str,
    epochs: list[int],
    columns: list[tuple[list[str], str]],
    groups: list[dict],
):
    for epoch in epochs:
        for column in columns:
            plot(epoch, title, True, column[0], groups, "Epoch", column[1])
            plot(epoch, title, False, column[0], groups, "Epoch", column[1])


if __name__ == "__main__":
    epochs = [750, 1000]

    groupA = [
        dict(
            label="Default",
            files=file_list(["final-a0-gt-virt"]),
        ),
        dict(
            label="Cond",
            files=file_list(["final-a2-gt-virt"]),
        ),
        dict(
            label="Sum",
            files=file_list(["final-a4-gt-virt"]),
        ),
        dict(
            label="Mem",
            files=file_list(["final-a6-gt-virt"]),
        ),
        dict(
            label="Rel",
            files=file_list(["final-a10-gt-virt"]),
        ),
        dict(
            label="Rel*",
            files=file_list(["final-a12-gt-virt"]),
        ),
    ]
    columns = [(["stats.success.rate"], "Success Rate")]
    plot_grid("standard", epochs, columns, groupA)

    groupB = [
        dict(
            label="Scene A",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene1"
            ),
        ),
        dict(
            label="Scene B",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene3"
            ),
        ),
        dict(
            label="Scene C",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene5"
            ),
        ),
        dict(
            label="Scene D",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene6"
            ),
        ),
        dict(
            label="Scene E",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene7"
            ),
        ),
        dict(
            label="Scene F",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene8"
            ),
        ),
        dict(
            label="Scene G",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene9"
            ),
        ),
        dict(
            label="Scene H",
            files=file_list(
                ["final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5"], "scene10"
            ),
        ),
    ]
    plot_grid("fed_a0", epochs, columns, groupB)

    groupX = [
        dict(
            label="Scene A",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene1"),
        ),
        dict(
            label="Scene B",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene3"),
        ),
        dict(
            label="Scene C",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene5"),
        ),
        dict(
            label="Scene D",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene6"),
        ),
        dict(
            label="Scene E",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene7"),
        ),
        dict(
            label="Scene F",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene8"),
        ),
        dict(
            label="Scene G",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene9"),
        ),
        dict(
            label="Scene H",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene10"),
        ),
    ]
    plot_grid("fed_a10", epochs, columns, groupX)

    groupC = [
        dict(
            label="Mu 0.0",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.0-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu0.0-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.0-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
        dict(
            label="Mu 0.01",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.01-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.01-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
        dict(
            label="Mu 0.1",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5",
                    # "final-a0-gt-virt-fed-k5-mu0.1-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.1-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
        dict(
            label="Mu 1.0",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu1.0-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu1.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
    ]
    columns = [(["fed.stats.success.rate"], "Success Rate")]
    plot_grid("fed_a0_mu", epochs, columns, groupC)

    groupD = [
        dict(
            label="Alpha 0.0",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu0.01-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.01-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.1-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu1.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
        dict(
            label="Alpha 0.5",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.0-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu0.0-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu1.0-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5",
                    # "final-a0-gt-virt-fed-k5-mu0.1-alpha0.5",
                ],
                "scene1",
            ),
        ),
    ]
    plot_grid("fed_a0_alpha", epochs, columns, groupD)

    groupE = [
        dict(
            label="k 1",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k1-mu0.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu0.01-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu1.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k1-mu0.1-lr0.0003-wd0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu1.0-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k1-mu0.0-alpha0.5",
                ],
                "scene1",
            ),
        ),
        dict(
            label="k 5",
            files=file_list(
                [
                    "final-a0-gt-virt-fed-k5-mu0.0-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu0.01-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01-alpha0.5",
                    # "final-a0-gt-virt-fed-k5-mu0.1-alpha0.5",
                    "final-a0-gt-virt-fed-k5-mu1.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.0-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.1-lr0.0003-wd0.01",
                    "final-a0-gt-virt-fed-k5-mu0.01-lr0.0003-wd0.01",
                ],
                "scene1",
            ),
        ),
    ]
    plot_grid("fed_a0_k", epochs, columns, groupE)

    groupY = [
        dict(
            label="Virtual",
            files=file_list(["final-a10-gt-virt"]),
        ),
        dict(
            label="Real",
            files=file_list(["final-a10-gt"]),
        ),
        dict(
            label="Virt (Fed)",
            files=file_list(["final-a10-gt-virt-fed-k1-mu0.1-alpha0.5"], "scene1"),
        ),
        dict(
            label="Real (Fed)",
            files=file_list(["final-a10-gt-fed-k1-mu0.1-alpha0.5"], "scene1"),
        ),
    ]
    columns = [(["fed.stats.success.rate", "stats.success.rate"], "Success Rate")]
    plot_grid("comp", [100], columns, groupY)
