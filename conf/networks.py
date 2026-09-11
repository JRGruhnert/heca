from heca.heca_gnn.network import ActorNetwork, CriticNetwork, Network

default = Network.Config()

interact = Network.Config(
    actor=ActorNetwork.Config(use_option_interaction=True),
)

timeline = Network.Config(
    actor=ActorNetwork.Config(use_timeline_memory=True),
)

both = Network.Config(
    actor=ActorNetwork.Config(
        use_timeline_memory=True,
        use_option_interaction=True,
    ),
)

separate = Network.Config(
    actor=ActorNetwork.Config(
        use_timeline_memory=True,
        use_option_interaction=True,
    ),
    critic=CriticNetwork.Config(),
)

concat = Network.Config(
    actor=ActorNetwork.Config(use_film=False),
)
concat_timeline = Network.Config(
    actor=ActorNetwork.Config(use_timeline_memory=True, use_film=False),
)

NETWORK_NAMES = [
    "default",
    "interact",
    "timeline",
    "both",
    "separate",
    "concat",
    "concat_timeline",
]
