"""Smoke test for the actor/critic split in ``heca_gnn/network.py``.

Checks that the two networks are genuinely independent rather than only
separately callable:

1. their parameter sets are disjoint, and together they are exactly the
   container's parameters (no parameter is dropped or shared),
2. a critic-only backward produces no gradient on any actor parameter and vice
   versa,
3. an optimizer built on ``critic_net.parameters()`` leaves the actor frozen,
4. the container's ``forward``/``evaluate`` keep the old shapes,
5. the memory conditioning is only read by a critic configured for it.

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl python scripts/diag_split_networks.py
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import copy
import inspect

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch

import conf.networks
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.heca_gnn.network import ActorNetwork, CriticNetwork, Network
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models

OK = "  ok  "


def named(net: Network, prefix: str) -> dict[str, torch.nn.Parameter]:
    """Parameters of one sub-network, keyed by their full container name."""
    return {
        k: p for k, p in net.named_parameters() if k.startswith(prefix + ".")
    }


def report(label: str, passed: bool, detail: str = ""):
    print(f"[{OK if passed else ' FAIL '}] {label}{f' — {detail}' if detail else ''}")
    assert passed, label


def main():
    torch.manual_seed(0)
    np.random.seed(0)

    # A real graph, so the shapes are the training shapes.
    scene = Scene.get(find_scene_config("scene0"), auto_load=False)
    cfgs = find_scene_models("scene0")
    for c in cfgs:
        e = ExpertModel.get(c, auto_load=False)
        e.use_gt(True)
        e.load()
        e.virtual()
    graph = Graph.generate(list(cfgs), smode=SubgoalMode.BOTH)
    (x, _), (y, _) = scene.sample_task()
    graph.set_goal(y)
    graph.set_start(x)
    data = graph.export()
    print(
        f"graph: {data['entity'].x.shape[0]} entity rows, "
        f"{data['option'].x.shape[0]} options, "
        f"{data['canonical'].x.shape[0]} canonical rows\n"
    )

    dim = conf.networks.both.actor.feature_dim

    # 1. parameter independence
    net = Network.get(conf.networks.both)
    a, c = named(net, "actor_net"), named(net, "critic_net")
    shared = set(a) & set(c)
    report("actor/critic parameter names are disjoint", not shared, f"shared={shared}")
    a_ids = {id(p) for p in a.values()}
    c_ids = {id(p) for p in c.values()}
    report(
        "no shared tensor between the two parameter sets",
        not (a_ids & c_ids),
        f"{len(a_ids)} actor + {len(c_ids)} critic params",
    )
    trunk_names = [
        n for n in ("trunc", "actor_trunc", "critic_trunc") if hasattr(net, n)
    ]
    trunk_ids: set[int] = set()
    for name in trunk_names:
        trunk_ids |= {id(p) for p in getattr(net, name).parameters()}
    report(
        "container parameters == actor + critic + trunk(s)",
        {id(p) for p in net.parameters()} == a_ids | c_ids | trunk_ids,
        f"trunk(s): {trunk_names}",
    )
    report(
        "trunk is " + ("separate per network" if len(trunk_names) > 1 else "shared"),
        True,
        f"seperate_trunc={net.cfg.seperate_trunc}",
    )
    report("every parameter is trainable", all(p.requires_grad for p in net.parameters()))

    # 2. gradients do not cross
    logits, value = net(data)
    report("container forward shapes", tuple(logits.shape) == (1, data["option"].x.shape[0]) and tuple(value.shape) == (1,), f"logits {tuple(logits.shape)}, value {tuple(value.shape)}")

    # Parameters outside the graph of this scene (an unused entity type) and
    # modules a call cannot touch (the GRU without ``mem_step``, the critic's
    # memory generator when no memory is passed) legitimately stay grad-free.
    def unused_only(names: list[str], allow: tuple[str, ...]) -> bool:
        return all(n.startswith(allow) for n in names)

    net.zero_grad(set_to_none=True)
    value.sum().backward()
    actor_touched = [k for k, p in a.items() if p.grad is not None]
    critic_untouched = [k for k, p in c.items() if p.grad is None]
    report(
        "critic-only backward leaves the actor untouched",
        not actor_touched,
        f"{len(c) - len(critic_untouched)}/{len(c)} critic params got grads",
    )
    report(
        "critic-only backward trains the critic",
        unused_only(
            critic_untouched,
            (
                "critic_net.encoder.entity_encoders.",
                "critic_net.encoder.comp_encoders.",
            ),
        ),
        f"untouched: {len(critic_untouched)} (entity types absent from the graph)",
    )

    net.zero_grad(set_to_none=True)
    (logits - logits.mean()).pow(2).sum().backward()
    actor_untouched = [k for k, p in a.items() if p.grad is None]
    critic_touched = [k for k, p in c.items() if p.grad is not None]
    report(
        "actor-only backward leaves the critic untouched",
        not critic_touched,
        f"{len(a) - len(actor_untouched)}/{len(a)} actor params got grads",
    )
    report(
        "actor-only backward trains the actor",
        unused_only(
            actor_untouched,
            (
                "actor_net.encoder.entity_encoders.",
                "actor_net.encoder.comp_encoders.",
                "actor_net.timeline_layer.",
            ),
        ),
        f"untouched: {len(actor_untouched)} (unused entity types / no mem_step)",
    )

    # 3. separate optimizers really are separate
    before_actor = {k: p.detach().clone() for k, p in a.items()}
    before_critic = {k: p.detach().clone() for k, p in c.items()}
    opt = torch.optim.SGD(net.critic_net.parameters(), lr=0.1)
    net.zero_grad(set_to_none=True)
    net.critic_net(data).sum().backward()
    opt.step()
    frozen_actor = [k for k, p in a.items() if not torch.equal(p.detach(), before_actor[k])]
    moved_critic = [k for k, p in c.items() if not torch.equal(p.detach(), before_critic[k])]
    report("a critic-only optimizer step leaves the actor frozen", not frozen_actor, f"moved={frozen_actor[:3]}")
    report("a critic-only optimizer step updates the critic", bool(moved_critic), f"{len(moved_critic)}/{len(c)} params moved")

    # 4. container evaluate + independent calls
    logprobs, values, entropies = net.evaluate([data, data], torch.tensor([0, 1]))
    report(
        "evaluate shapes",
        tuple(logprobs.shape) == (2,) and tuple(values.shape) == (2,) and tuple(entropies.shape) == (2,),
        f"{tuple(logprobs.shape)} {tuple(values.shape)} {tuple(entropies.shape)}",
    )
    report(
        "actor() == actor_net(), critic() == critic_net()",
        torch.allclose(net.actor(data), net.actor_net(data))
        and torch.allclose(net.critic(data), net.critic_net(data)),
    )

    # 5. the actor resolves the timeline memory; the critic never reads it
    data.mem_step = (torch.zeros(1, dim), torch.zeros(1, dim))
    net(data)
    used = net.actor_net._last_mem
    report(
        "the actor resolves a timeline memory",
        tuple(used.shape) == (1, dim) and bool(used.any()),
    )

    # build the config here rather than trusting a preset: the point is that the
    # two widths are independent
    separate = Network.get(
        Network.Config(
            actor=conf.networks.separate.actor,
            critic=CriticNetwork.Config(feature_dim=128),
        )
    )
    report(
        "critic feature_dim is independent of the actor",
        separate.critic_net.encoder.dim == 128 and separate.actor_net.encoder.dim == dim,
        f"actor {separate.actor_net.encoder.dim}, critic {separate.critic_net.encoder.dim}",
    )


    report(
        "the critic has no memory input at all",
        "memory" not in inspect.signature(separate.critic_net.forward).parameters
        and "memory" not in separate.critic_net.condenser_names,
    )

    # 6. checkpoints round-trip through the split key space
    state = net.state_dict()
    report(
        "state_dict is namespaced per network",
        all(k.startswith(("actor_net.", "critic_net.")) for k in state),
        f"{len(state)} tensors",
    )
    fresh = Network.get(
        Network.Config(actor=copy.deepcopy(conf.networks.both.actor), critic=copy.deepcopy(conf.networks.both.critic))
    )
    fresh.load_state_dict(state)
    report(
        "state_dict round-trips",
        all(torch.equal(v, fresh.state_dict()[k]) for k, v in state.items()),
    )

    # 7. all presets still build
    built = []
    for name in conf.networks.NETWORK_NAMES:
        n = Network.get(getattr(conf.networks, name))
        built.append(f"{name}({sum(p.numel() for p in n.actor_net.parameters())}a/"
                     f"{sum(p.numel() for p in n.critic_net.parameters())}c)")
    print(f"[{OK}] all presets build: {', '.join(built)}")

    # 8. standalone construction, independent of the container
    actor = ActorNetwork.get(ActorNetwork.Config(use_option_effects=True))
    critic = CriticNetwork.get(CriticNetwork.Config(feature_dim=64))
    report(
        "actor and critic run standalone",
        tuple(actor(data).shape) == (1, data["option"].x.shape[0]) and tuple(critic(data).shape) == (1,),
    )

    print("\nall split checks passed")


if __name__ == "__main__":
    main()
