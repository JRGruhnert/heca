"""Why the value-loss MSE warning fires, and what it does to training.

The warning is

    Using a target size (torch.Size([69, 2048])) that is different to the input
    size (torch.Size([69, 1]))

with minibatch 69 and buffer capacity 2048. This script reproduces the shape
chain end to end on a real graph:

1. what the critic returns per step (``Network.forward``), and where the extra
   dimension comes from (``FiLMStack`` broadcasting a ``[1, D]`` conditioning
   input against a ``[D]`` activation),
2. what ``FairBuffer._gae_for_bucket`` does with that shape (returns become
   ``[T, T]`` instead of ``[T]``),
3. what ``MSELoss`` then optimizes (each prediction against ``T`` targets, so the
   optimum is ``advantage_i + mean_j value_j`` instead of ``advantage_i +
   value_i``),
4. which rows of the broken target are still correct (only the diagonal).

Usage:
    PYG_HOME=/tmp/pyg_cache MPLCONFIGDIR=/tmp/mpl python scripts/diag_value_shape.py
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, _REPO_ROOT)

import matplotlib

matplotlib.use("Agg")

import numpy as np
import torch
from torch import nn

import conf.networks
from heca.experts.expert import ExpertModel
from heca.graphs.graph import Graph, SubgoalMode
from heca.heca_gnn.network import Network
from heca.learning.buffers.fair_buffer import FairBuffer
from heca.scenes.scene import Scene
from scripts.common.scenes import find_scene_config, find_scene_models


def main():
    torch.manual_seed(0)
    np.random.seed(0)

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
    data = graph.build()

    net = Network.get(conf.networks.default)
    net.eval()
    with torch.inference_mode():
        logits, value = net(data)
    print(f"actor logits shape          : {tuple(logits.shape)}")
    print(f"critic value shape per step : {tuple(value.shape)}")

    # 1. where the dimension is added
    dim = net.cfg.actor.feature_dim
    films = net.critic_net.films
    stats = net.critic_net.state_critic.last_stats
    print(f"\nStateCritic.last_stats       : {tuple(stats.shape)}")
    conds = {"goal": torch.zeros(1, dim)}
    site = net.critic_net.state_critic.SITE
    print(
        f"FiLMStack(['D'], conds)      : {tuple(films(torch.zeros(dim), conds, site).shape)}"
    )
    print(
        "  -> a [1, D] conditioning input broadcasts a [D] activation to [1, D],"
        "\n     and the critic's final Linear turns that into [1, 1] instead of [1]."
    )

    # 2. what the buffer does with [1, 1] values
    T = 8
    rewards = [0.0, 1.0, 0.0, 0.0, -0.01, 0.0, 0.0, 0.0]
    terminals = [False] * 7 + [True]
    vals = torch.linspace(-0.5, 0.5, T)

    buf = FairBuffer.__new__(FairBuffer)
    buf.cfg = FairBuffer.Config()
    adv_2d, rtn_2d = buf._gae_for_bucket(
        rewards, terminals, [v.reshape(1, 1) for v in vals]
    )
    adv_1d, rtn_1d = buf._gae_for_bucket(
        rewards, terminals, [v.reshape(1) for v in vals]
    )
    print(
        f"\nvalues per step [1, 1] -> advantages {tuple(adv_2d.shape)}, returns {tuple(rtn_2d.shape)}"
    )
    print(
        f"values per step [1]    -> advantages {tuple(adv_1d.shape)}, returns {tuple(rtn_1d.shape)}"
    )
    print(
        f"returns agree on the diagonal: {bool(torch.allclose(torch.diagonal(rtn_2d), rtn_1d))}"
    )
    print(f"max |returns_2d - correct|  : {float((rtn_2d - rtn_1d).abs().max()):.4f}")

    # 3./4. what MSELoss optimizes against the [B, T] target
    B = 4
    target = rtn_2d[:B]  # (B, T) as in the minibatch
    pred = torch.zeros(B, 1, requires_grad=True)
    loss = nn.MSELoss()(pred, target)
    loss.backward()
    optimum = target.mean(dim=1)
    correct = rtn_1d[:B]
    print(
        f"\nloss(...) shape                 : {tuple(loss.shape) if loss.dim() else 'scalar'}"
    )
    print(f"gradient w.r.t. prediction      : {pred.grad.flatten().tolist()}")
    print(f"optimum (mean over T targets)   : {optimum.tolist()}")
    print(f"correct value target            : {correct.tolist()}")
    print(f"systematic offset (mean - true) : {(optimum - correct).tolist()}")
    print(
        "\nThe offset is value_i - mean_j value_j: the broadcast target removes\n"
        "exactly the per-state value variation the critic is supposed to fit."
    )


if __name__ == "__main__":
    main()
