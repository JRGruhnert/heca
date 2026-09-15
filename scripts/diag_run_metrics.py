"""Metric history of local wandb runs, read from their ``output.log``.

wandb in online mode keeps the history on the server, but the console output of
every run is written locally, and that is where ``Update N | k=v, ...`` lines
live. This script turns them back into series:

    python scripts/diag_run_metrics.py                 # one row per run
    python scripts/diag_run_metrics.py wandb/run-...   # time windows of one run

Only runs that actually logged ``expl_var`` and ran for >=100 updates are shown.
Sorting is by the explained variance of the last 20 updates, so the "critic looks
good" runs are on top.
"""

import re
import sys
from pathlib import Path

import numpy as np

LOG = "files/output.log"


def series(log: Path) -> list[tuple[int, dict[str, float]]]:
    rows: list[tuple[int, dict[str, float]]] = []
    for line in log.read_text(errors="ignore").splitlines():
        m = re.search(r"Update\s+(\d+) \| (.+)$", line)
        if not m:
            continue
        values: dict[str, float] = {}
        for part in m.group(2).split(", "):
            if "=" in part:
                key, val = part.split("=", 1)
                try:
                    values[key.strip()] = float(val)
                except ValueError:
                    pass
        if values:
            rows.append((int(m.group(1)), values))
    return rows


def window(rows, lo: int, hi: int, key: str) -> float:
    vals = [v.get(key, np.nan) for n, v in rows if lo <= n < hi]
    return float(np.nanmean(vals)) if vals else float("nan")


def one_run(run_dir: Path) -> None:
    rows = series(run_dir / LOG)
    if not rows:
        print(f"{run_dir.name}: no metric lines")
        return
    n = rows[-1][0]
    print(f"{run_dir.name}: {n} updates")
    print(f"  {'updates':>14s} {'expl_var':>9s} {'success':>8s} {'entropy':>8s} "
          f"{'approx_kl':>10s} {'policy_loss':>12s} {'value_loss':>11s}")
    cuts = [1, n // 3, 2 * n // 3, max(n - 40, 1)]
    for lo in cuts:
        print(
            f"  {lo:5d}-{min(lo + 40, n):5d} "
            + " ".join(
                f"{window(rows, lo, lo + 40, k):>9.4f}"
                for k in ("expl_var", "stats/success_rate", "entropy", "approx_kl")
            )
            + f" {window(rows, lo, lo + 40, 'policy_loss'):>+12.4f}"
            + f" {window(rows, lo, lo + 40, 'value_loss'):>11.4f}"
        )


def sweep() -> None:
    out = []
    for log in sorted(Path("wandb").glob("run-*/" + LOG)):
        if "expl_var" not in log.read_text(errors="ignore"):
            continue
        rows = series(log)
        if len(rows) < 100:
            continue
        ev = np.array([v.get("expl_var", np.nan) for _, v in rows])
        sr = np.array([v.get("stats/success_rate", np.nan) for _, v in rows])
        keep = ~np.isnan(ev) & ~np.isnan(sr)
        lo = rows[-20][0]
        hi = rows[-1][0] + 1
        out.append(
            (
                log.parent.parent.name,
                len(rows),
                float(np.nanmean([v.get("expl_var", np.nan) for _, v in rows[-20:]])),
                float(
                    np.nanmean(
                        [v.get("stats/success_rate", np.nan) for _, v in rows[-20:]]
                    )
                ),
                float(
                    np.nanmean(
                        [v.get("stats/mean_length", np.nan) for _, v in rows[-20:]]
                    )
                ),
                float(np.nanmean([v.get("entropy", np.nan) for _, v in rows[-20:]])),
                float(np.corrcoef(ev[keep], sr[keep])[0, 1]) if keep.sum() > 1 else np.nan,
                lo,
                hi,
            )
        )
    out.sort(key=lambda r: -r[2])
    print(f"{'run':32s} {'n':>5s} {'ev_last':>8s} {'succ_last':>9s} "
          f"{'len_last':>8s} {'ent_last':>8s} {'corr':>6s}")
    for r in out:
        print(f"{r[0]:32s} {r[1]:5d} {r[2]:8.3f} {r[3]:9.3f} {r[4]:8.2f} "
              f"{r[5]:8.3f} {r[6]:+6.2f}")
    print(f"\nruns scanned: {len(out)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        one_run(Path(sys.argv[1]))
    else:
        sweep()
