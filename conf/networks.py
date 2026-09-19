from heca.heca_gnn.network import Network

# virtual
a0 = Network.Config()
a1 = Network.Config(goal_conditioning="hyperedge")
a2 = Network.Config(use_condition_gat=True)
a3 = Network.Config(goal_conditioning="hyperedge", use_condition_gat=True)
a4 = Network.Config(use_summary_gcn=True)
a5 = Network.Config(goal_conditioning="hyperedge", use_summary_gcn=True)
a6 = Network.Config(use_memory=True)
a7 = Network.Config(goal_conditioning="hyperedge", use_memory=True)
a8 = Network.Config(use_statistics=True)
a9 = Network.Config(goal_conditioning="hyperedge", use_statistics=True)

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


NETWORK_NAMES = ["a0", "a1", "a2", "a3", "a4", "a5", "a6", "a7", "a8", "a9"]

CONFIGS: dict[str, Network.Config] = {name: globals()[name] for name in NETWORK_NAMES}


def get(name: str) -> Network.Config:
    """The config a ``--network`` name refers to, curated or derived."""
    if name not in CONFIGS:
        raise KeyError(
            f"unknown network config {name!r}; known: {', '.join(NETWORK_NAMES)}"
        )
    return CONFIGS[name]
