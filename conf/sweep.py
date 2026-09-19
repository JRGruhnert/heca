from itertools import product
from typing import Any

from heca.heca_gnn.network import Network

# flag -> (code, values); the first value is that axis' baseline.
AXES: dict[str, tuple[str, tuple[Any, ...]]] = {
    "use_condition_gat": ("c", (False, True)),
    "use_summary_gcn": ("s", (False, True)),
    "goal_conditioning": ("g", ("none", "hyperedge", "residual")),
    "jitter_scope": ("j", ("none", "entity", "scene", "both")),
    "use_option_transformer": ("x", (False, True)),
}

BASE_NAME = "base"


def _token(code: str, value: Any) -> str:
    if value is True:
        return code
    if value is False:
        return f"{code}0"
    return f"{code}{value}"


def name_of(values: dict[str, Any]) -> str:
    """Sweep name of one combination: the codes of the non-baseline axes."""
    tokens = [
        _token(code, values[flag])
        for flag, (code, vals) in AXES.items()
        if values[flag] != vals[0]
    ]
    return "".join(tokens) or BASE_NAME


def combos() -> dict[str, Network.Config]:
    flags = list(AXES)
    out: dict[str, Network.Config] = {}
    for values in product(*(AXES[flag][1] for flag in flags)):
        picked = dict(zip(flags, values))
        name = name_of(picked)
        if name in out:
            raise ValueError(f"two combinations share the name {name!r}: {picked}")
        out[name] = Network.Config(**picked)
    return out


SWEEP: dict[str, Network.Config] = combos()


def flags_of(name: str) -> dict[str, Any]:
    """The non-baseline flag values of one sweep entry."""
    cfg = SWEEP[name]
    return {
        flag: value
        for flag, (_, vals) in AXES.items()
        if (value := getattr(cfg, flag)) != vals[0]
    }


# base first, then fewest changes first, so the table reads as a progression
SWEEP_NAMES: list[str] = sorted(SWEEP, key=lambda name: (len(flags_of(name)), name))


def _print_table(with_params: bool = False) -> None:
    header = ["name", *(f"{f.removeprefix('use_')}[{c}]" for f, (c, _) in AXES.items())]
    if with_params:
        header.append("params")

    rows: list[list[str]] = []
    for name in SWEEP_NAMES:
        picked = flags_of(name)
        row = [name, *(str(picked.get(flag, ".")) for flag in AXES)]
        if with_params:
            try:
                net = Network(SWEEP[name])
                row.append(f"{sum(p.numel() for p in net.parameters()):,}")
            except Exception as exc:  # e.g. an ambiguous memory/root combination
                row.append(f"!! {type(exc).__name__}")
        rows.append(row)

    widths = [
        max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))
    ]
    print("  ".join(h.ljust(w) for h, w in zip(header, widths)))
    print("  ".join("-" * w for w in widths))
    for row in rows:
        print("  ".join(c.ljust(w) for c, w in zip(row, widths)))
    print(f"\n{len(SWEEP_NAMES)} configs from axes: {', '.join(AXES)}")
    print("'.' = baseline value, otherwise the listed value.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--params",
        action="store_true",
        help="Build each config and add its parameter count (flags invalid ones).",
    )
    _print_table(parser.parse_args().params)
