"""Compare c05_explain runs in one figure, one column per run.

Each column is exactly the figure ``c05_explain`` already draws -- option logits
(options x steps, viridis) above entity saliency (entities x steps, RdYlGn) --
and the runs are compared by putting their columns next to each other.  The two
panels keep the height ratio of their row counts, so the 40-option logits
heatmap and the 5-entity saliency heatmap stay readable in one figure.

The entity names are drawn only under the leftmost column (``--ylabels left``,
the default): printing them under every column is what forces a large
horizontal gap between the columns.

    python -m scripts.d09_explain_compare \
        --explain a10=data/network/standard/scene0_final-a10-gt-virt_s1/explain/ckp_1000_a10 \
        --explain a12=data/network/standard/scene0_final-a10-gt-virt_s1/explain/ckp_1000_a12 \
        --episode 0 --out plots/explain_compare_ep00.png
"""

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from heca.misc.paths import output_root
from scripts.c05_explain import plot_compare


def find_episode(directory: Path, episode: int, checkpoint: str | None) -> Path:
    """``<dir>/*ep{NN}.json`` for the requested checkpoint, newest if ambiguous."""
    pattern = f"{checkpoint or '*'}_ep{episode:02d}.json"
    matches = sorted(directory.glob(pattern))
    if not matches:
        available = sorted(p.name for p in directory.glob("*_ep*.json"))
        raise SystemExit(
            f"no {pattern!r} in {directory}"
            + (f"; found {available[:5]}" if available else "")
        )
    return matches[-1]


def load(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not payload.get("explanations"):
        raise SystemExit(f"{path} has no explained steps")
    return payload


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--explain",
        action="append",
        default=[],
        metavar="TAG=DIR",
        required=True,
        help="Explain directory to draw, in drawing order (repeatable).",
    )
    ap.add_argument("--episode", type=int, default=0, help="Episode number (0).")
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="Checkpoint prefix of the explain files, e.g. ckp_1000 (default: any).",
    )
    ap.add_argument(
        "--rows",
        choices=["both", "logits", "saliency"],
        default="both",
        help="Panels drawn per column: logits above saliency (default), or only "
        "one of them.",
    )
    ap.add_argument(
        "--ylabels",
        choices=["left", "all", "none"],
        default="left",
        help="Where the entity names go: only the leftmost column (default), "
        "every column, or nowhere.",
    )
    ap.add_argument(
        "--y-options",
        action="store_true",
        help="Also label the option names on the y axis of the leftmost column's "
        "logits panel.",
    )
    ap.add_argument(
        "--shared-scale",
        action="store_true",
        help="One colour scale for every saliency panel, so colours compare "
        "across runs (default: each column scales to its own max).",
    )
    ap.add_argument(
        "--out",
        default=str(output_root() / "plots/explain_compare.png"),
        help="Output PNG (default: <repo or $HECA_OUT>/plots/explain_compare.png).",
    )
    ap.add_argument("--n-steps", type=int, default=16, help="Steps shown (16).")
    ap.add_argument("--dpi", type=int, default=200, help="Output dpi (200).")
    args = ap.parse_args()

    columns = []
    for entry in args.explain:
        tag, _, directory = entry.partition("=")
        if not directory:
            raise SystemExit(f"--explain expects TAG=DIR, got {entry!r}")
        columns.append((tag, load(find_episode(Path(directory), args.episode, args.checkpoint))))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    plot_compare(
        out,
        columns,
        rows=args.rows,
        ylabels=args.ylabels,
        n_steps=args.n_steps,
        shared_scale=args.shared_scale,
        y_options=args.y_options,
        dpi=args.dpi,
    )
    print(f"wrote {out}  ({len(columns)} column(s): {', '.join(t for t, _ in columns)})")


if __name__ == "__main__":
    main()
