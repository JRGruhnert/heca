from heca.heca_gnn.network import Network

# NETWOR ABLATIONS
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

# FEDERATED ONLY ABLATIONS
x0 = Network.Config(sync=("!actor_head", "!critic_head"))  # federate trunk
x1 = Network.Config(sync=("root",))  # federate only root

NETWORK_NAMES = [
    "a0",
    "a1",
    "a2",
    "a3",
    "a4",
    "a5",
    "a6",
    "a7",
    "a8",
    "a9",
    "x0",
    "x1",
]

CONFIGS: dict[str, Network.Config] = {name: globals()[name] for name in NETWORK_NAMES}


def get(name: str) -> Network.Config:
    """The config a ``--network`` name refers to, curated or derived."""
    if name not in CONFIGS:
        raise KeyError(
            f"unknown network config {name!r}; known: {', '.join(NETWORK_NAMES)}"
        )
    return CONFIGS[name]
