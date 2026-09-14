"""Network configurations for the ablation studies.

One entry per ablation axis, so a run is ``--network <name>``. Layer choices
that only swap a module (condition, summary, scene, hyperedge) live here; which
of them is federated lives in ``FLServer.Config.sync_layers`` (``--sync``).
"""

from heca.heca_gnn.network import Network

# no ablation, every optional layer is its identity / GIN / GCN default
default = Network.Config()

# --- single layer swaps -------------------------------------------------
condition_gat = Network.Config(use_condition_gat=True)
summary_gcn = Network.Config(use_summary_gcn=True)
hyperedge = Network.Config(use_hyperedge=True)
effects = Network.Config(use_option_effects=True)

# --- optional stages ----------------------------------------------------
interact = Network.Config(use_option_transformer=True)
timeline = Network.Config(use_memory=True)
both = Network.Config(use_option_transformer=True, use_memory=True)

# --- combinations -------------------------------------------------------
# per-role trunc, shared root
separate = Network.Config(
    seperate_trunc=True, use_option_transformer=True, use_memory=True
)
# per-role root, everything after it shared: the two roles see different entity
# rows but share the option/scene path (memory would be ambiguous here, so it
# stays off)
separate_root = Network.Config(seperate_root=True)
# both segments per role
split = Network.Config(
    seperate_root=True, seperate_trunc=True, use_option_transformer=True, use_memory=True
)
full = Network.Config(
    use_option_transformer=True,
    use_condition_gat=True,
    use_summary_gcn=True,
    use_memory=True,
    use_hyperedge=True,
    use_option_effects=True,
)

NETWORK_NAMES = [
    "default",
    "condition_gat",
    "summary_gcn",
    "hyperedge",
    "effects",
    "interact",
    "timeline",
    "both",
    "separate",
    "separate_root",
    "split",
    "full",
]
