from heca.heca_gnn.network import Network

free = Network.Config(gating=False)  # doesnt learn virtual

# TODO: need greedy search scene0 virtual evaluation

# virtual
a0 = Network.Config()
a1 = Network.Config(seperate_root=True, seperate_trunc=True)
a2 = Network.Config(seperate_root=False, seperate_trunc=True)

# on best of a
b0 = Network.Config(seperate_root=False, seperate_trunc=True)
b1 = Network.Config(seperate_root=False, seperate_trunc=True)
b2 = Network.Config(seperate_root=False, seperate_trunc=True)
b3 = Network.Config(seperate_root=False, seperate_trunc=True)
b4 = Network.Config(seperate_root=False, seperate_trunc=True)

# on best model with noise (0.0 ,0.2, 0.4, 0.6)
c0 = Network.Config()

# with (real models) GT
d0 = Network.Config()

# with (real models) VISUAL
e0 = Network.Config()

# FEDERATED

# special on visual (best of all runs at the end)
# in Network.Config
sync = ()  # federate everything (baseline)
sync = ("!actor_head", "!critic_head")  # federate trunk, keep heads local
sync = ("root",)  # federate only the encoder/root
sync = ("!critic_head",)  # keep the value head local only


NETWORK_NAMES = [
    "a0",
    "a1",
    "a2",
]

CONFIGS: dict[str, Network.Config] = {name: globals()[name] for name in NETWORK_NAMES}


def get(name: str) -> Network.Config:
    """The config a ``--network`` name refers to, curated or derived."""
    if name not in CONFIGS:
        raise KeyError(
            f"unknown network config {name!r}; known: {', '.join(NETWORK_NAMES)}"
        )
    return CONFIGS[name]
