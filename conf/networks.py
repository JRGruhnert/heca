from heca.heca_gnn.network import Network

from conf.sweep import SWEEP, SWEEP_NAMES

default = Network.Config()
free = Network.Config(gating=False)
sage = Network.Config(use_summary_gcn=True)
separate = Network.Config(seperate_root=True, seperate_trunc=True)
separate_trunc = Network.Config(seperate_root=False, seperate_trunc=True)

best = Network.Config(
    seperate_root=False,
    seperate_trunc=True,
    use_option_transformer=True,
    use_condition_gat=True,
    use_summary_gcn=True,
    use_memory=True,
    use_hyperedge=True,
)
CURATED_NAMES = [
    "default",
    "sage",
    "free",
    "separate",
    "separate_trunc",
]

# every derived sweep config (conf/sweep.py) is a network config too, so
# ``--network <sweep name>`` works like any curated one
CONFIGS: dict[str, Network.Config] = {
    name: globals()[name] for name in CURATED_NAMES
} | SWEEP

_clashes = set(CURATED_NAMES) & set(SWEEP_NAMES)
assert not _clashes, f"derived sweep names clash with curated ones: {sorted(_clashes)}"

NETWORK_NAMES = [*CURATED_NAMES, *SWEEP_NAMES]


def get(name: str) -> Network.Config:
    """The config a ``--network`` name refers to, curated or derived."""
    if name not in CONFIGS:
        raise KeyError(
            f"unknown network config {name!r}; known: {', '.join(NETWORK_NAMES)}"
        )
    return CONFIGS[name]


def __getattr__(name: str) -> Network.Config:
    """Derived sweep configs as module attributes (``conf.networks.csx``)."""
    if name in SWEEP:
        return SWEEP[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
