import argparse, json, sys, time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib

matplotlib.use("Agg")
import numpy as np, torch

import conf.networks
from heca.agents.heca import Heca
from heca.data.entity import Entity
from heca.graphs.graph import SubgoalMode
from heca.learning.ppo import PPO
from heca.misc import logger
from scripts.common.scenes import find_scene_models

ap = argparse.ArgumentParser()
ap.add_argument("--network", default="x0")
ap.add_argument("--rotation", default="cfg")
ap.add_argument("--scene", default="scene0")
ap.add_argument("--batch", type=int, default=120)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--env-seed", type=int, default=12345)
ap.add_argument("--eval-every", type=int, default=20)
ap.add_argument("--eval-tasks", type=int, default=16)
ap.add_argument("--out", default="")
ap.add_argument(
    "--max-steps",
    type=int,
    default=0,
    help="override Scene.Config.max_steps (0 keeps the default)",
)
ap.add_argument(
    "--terms",
    default="default",
    choices=("default", "with-z2"),
    help="'with-z2' restores the pre-drop edge terms",
)
ap.add_argument(
    "--buggy-projection",
    action="store_true",
    help="restore the pre-fix Entity.project (destination offsets), "
    "i.e. reproduce the runs that did not learn",
)
args = ap.parse_args()

if args.buggy_projection:
    # the old projection: slices the *destination* block offsets out of the
    # source tensor, so the extra/joint block receives orientation columns
    def old_project(cls, features, dst_layout):
        chunks = []
        for _name, block in dst_layout.items():
            chunks.append(features[:, block.mean()])
            if block.logstd_dim:
                chunks.append(features[:, block.logstd()])
        return (
            torch.cat(chunks, dim=-1)
            if isinstance(features, torch.Tensor)
            else np.concatenate(chunks, axis=-1)
        )

    Entity.project = classmethod(old_project)
    print("using the pre-fix Entity.project", flush=True)

torch.manual_seed(args.seed)
np.random.seed(args.seed)
net_cfg = conf.networks.get(args.network)
if args.rotation != "cfg":
    net_cfg = replace(net_cfg, use_rotation=args.rotation == "on")
if args.terms == "with-z2":
    net_cfg = replace(net_cfg, edge_terms=("z", "z2", "logz", "logp", "state"))

arm = (
    f"{args.network}_rot-{net_cfg.use_rotation}"
    + (f"_steps{args.max_steps}" if args.max_steps else "")
    + ("" if args.terms == "default" else f"_{args.terms}")
    + ("_buggyproj" if args.buggy_projection else "")
)
heca = Heca.get(
    Heca.Config(
        experts=find_scene_models(args.scene),
        learner=PPO.Config(
            tag=f"ab_{arm}",
            group="ab",
            network=net_cfg,
            label="diag",
            wandb=logger.WandBConfig(enabled=False),
            max_update=args.batch,
            lr_annealing=False,
        ),
        visualize=False,
        inference=False,
        reload=False,
        smode=SubgoalMode.BOTH,
        virtual=True,
        use_gt=True,
    )
)
if args.max_steps:
    heca.scene.cfg.max_steps = args.max_steps
heca.scene._env.reset(seed=args.env_seed)  # same task sequence in both arms
learner = heca.learner
tasks = list(range(1, args.eval_tasks + 1))
rng = np.random.RandomState(args.seed)
random_predict = lambda data: int(
    rng.choice(np.flatnonzero(data.option.gated.numpy() == 0))
)


def evaluate(random_baseline: bool) -> float:
    """Greedy (or uniform-random) policy success on the fixed task ids."""
    learner.eval()
    saved = learner.predict
    if random_baseline:
        learner.predict = random_predict  # type: ignore[method-assign]
    hits = 0
    for task in tasks:
        heca.scene.seed = 1000 + task
        x, y = heca.sample()
        _, fb = heca.act(x, y)
        hits += int(fb.success)
    learner.predict = saved  # type: ignore[method-assign]
    learner.train_mode = True
    heca.scene.seed = None
    return hits / len(tasks)


records = []
start = time.perf_counter()
updates = 0
if args.eval_every:
    records.append(dict(update=0, greedy=evaluate(False), random=evaluate(True)))
    print(
        f"[{arm}] update   0 greedy={records[-1]['greedy']:.3f} random={records[-1]['random']:.3f} "
        f"({time.perf_counter() - start:.0f}s)",
        flush=True,
    )
while updates < args.batch:
    if not heca.tick():
        continue
    updates += 1
    # mirrors scripts/c02_train_seq.py: push the trained weights into the frozen
    # behaviour copy the rollout samples from, otherwise every arm rolls out with
    # its random initialisation and no arm can learn
    learner.sync()
    m = learner.metrics
    if args.eval_every and updates % args.eval_every == 0:
        g = evaluate(False)
        records.append(
            dict(
                update=updates,
                greedy=g,
                random=records[-1]["random"],
                success=m.get("stats/success_rate"),
                entropy=m.get("train/entropy"),
                expl_var=m.get("train/expl_var"),
                kl=m.get("train/approx_kl"),
            )
        )
        print(
            f"[{arm}] update {updates:3d} greedy={g:.3f} train_success={m.get('stats/success_rate', float('nan')):.3f} "
            f"entropy={m.get('train/entropy', float('nan')):.3f} expl_var={m.get('train/expl_var', float('nan')):.3f} "
            f"({time.perf_counter() - start:.0f}s)",
            flush=True,
        )
    elif updates % 5 == 0:
        print(
            f"[{arm}] update {updates:3d} train_success={m.get('stats/success_rate', float('nan')):.3f} "
            f"entropy={m.get('train/entropy', float('nan')):.3f}",
            flush=True,
        )

if args.out:
    Path(args.out).write_text(json.dumps(records, indent=1))
print(
    f"[{arm}] done: {updates} updates in {time.perf_counter() - start:.0f}s", flush=True
)
