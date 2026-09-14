"""Toy: does a hyperedge over "rows of the same entity" actually deliver identity?

Throwaway experiment for the identity/correspondence question. No project code
is touched: the graph is synthetic and the models are plain tensor ops.

Task (per sample):
  - ``E`` entities, each with a required change ``delta[e]`` (all non-zero,
    distinct directions) from its current to its goal value.
  - ``O`` options; option ``o`` targets entity ``e_o = o % E`` and *proposes* a
    change on it. All proposals are copies of some entity's ``delta``, so the
    magnitudes are uninformative; exactly one option proposes the change that
    its own target entity actually requires.
  - The model must pick that option. Solving it requires pairing an option's
    proposal with the (current, goal) rows *of the same entity*; matching against
    a global pool of rows cannot work, because the required change of an entity
    is only visible as goal - current of that entity.

Variants (identical capacity/optimizer, only the information routing differs):
  bag                rows carry role embeddings, everything is pooled globally
                     (identity absent, ordering only as tags)
  global_role_pool   like the plain "global state slots": per-role means over all
                     entities, no per-entity grouping
  mean_hub           one hyperedge per entity, pooled to a single vector
                     (identity present, ordering lost inside the hub)
  slots_pooled       role-indexed slots per entity, but the consumer receives the
                     pooled hub (ordering lost at the consumer)
  role_slots         role-indexed slots per entity, consumer compares its own
                     proposal against that entity's slots (identity + ordering)

Usage:
    python scripts/toy_hyperedge_identity.py
"""

import argparse

import torch
from torch import nn

E, O, D = 4, 8, 16
ROLES = ("cur", "goal", "post")
R = len(ROLES)


def make_batch(batch: int, seed: int | None = None):
    """Returns cur (B,E,D), goal (B,E,D), prop (B,O,E,D), label (B,)."""
    g = torch.Generator().manual_seed(seed) if seed is not None else None

    def rand(*shape):
        v = torch.randn(*shape, generator=g)
        return v / v.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    cur = 0.5 * torch.randn(batch, E, D, generator=g)
    delta = rand(batch, E, D)  # every entity requires a change
    goal = cur + delta

    target = torch.arange(O) % E  # entity each option targets
    correct = torch.randint(0, O, (batch,), generator=g)  # random answer

    # option o proposes the change of entity (o+1) % E, except the correct one,
    # which proposes its own target's change -> exactly one match per sample.
    src = (torch.arange(O) + 1) % E
    prop = torch.zeros(batch, O, E, D)
    for o in range(O):
        e = int(target[o])
        matches = correct == o
        src_here = delta[:, int(src[o])]  # wrong entity's change
        src_here = torch.where(matches[:, None], delta[:, e], src_here)
        prop[:, o, e] = src_here
    return cur, goal, prop, correct


class MLP(nn.Module):
    def __init__(self, inp: int, hidden: int = 128, out: int | None = None):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(inp, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, out if out is not None else hidden),
        )

    def forward(self, x):
        return self.net(x)


class Toy(nn.Module):
    """One comparison head, six routings. The only difference between variants is
    *what* the consumer receives; the matching itself is a shared learned metric
    ``-||W(a) - W(b)||^2`` so that optimisation luck cannot masquerade as routing.

    Variants differ in how the per-entity pairing information reaches the
    consumer:

    paired          consumer gets, per entity, its own proposal and (goal, cur)
                    -> identity + ordering, explicitly paired
    role_slots      consumer gets, per entity, its own proposal and the entity's
                    role-indexed hyperedge slots (cur / goal)
    mean_hub        consumer gets, per entity, its own proposal and the *pooled*
                    hub (cur and goal already averaged, ordering gone)
    slots_pooled    consumer gets its proposal pooled over entities plus the slot
                    set pooled over entities and roles (no pairing, no ordering)
    global_role_pool  consumer gets per-role means over all entities (no pairing)
    bag             consumer gets one global mixed pool (no roles, no pairing)
    """

    def __init__(self, variant: str):
        super().__init__()
        self.variant = variant
        self.role_emb = nn.Embedding(R, D)
        self.hub_up = MLP(D + D, out=D)  # (member, role) -> additive slot update
        self.metric = nn.Linear(D, D, bias=False)  # shared comparison metric
        self.pool = MLP(2 * D, out=D)  # maps a pooled context back to D

    def hyperedges(self, cur_r, goal_r, prop_r):
        """One hyperedge per entity. Content is kept, the hub only adds."""
        B = cur_r.shape[0]
        role = torch.arange(R)
        if self.variant == "mean_hub":
            members = torch.stack([cur_r, goal_r], 2)  # (B,E,2,D)
            attr = self.role_emb(role[:2])[None, None].expand(B, E, 2, D)
            upd = self.hub_up(torch.cat([members, attr], -1))
            return (members + upd).mean(2)  # (B,E,D) — cur/goal already pooled
        post_slot = prop_r.mean(1)  # all options' proposals, per entity
        members = torch.stack([cur_r, goal_r, post_slot], 2)  # (B,E,R,D)
        attr = self.role_emb(role)[None, None].expand(B, E, R, D)
        upd = self.hub_up(torch.cat([members, attr], -1))
        return members + upd  # (B,E,R,D)

    def forward(self, cur, goal, prop):
        B = cur.shape[0]
        role = torch.arange(R)
        cur_r = cur + self.role_emb(role[0])
        goal_r = goal + self.role_emb(role[1])
        prop_r = prop + self.role_emb(role[2])  # (B,O,E,D)
        prop_pool = prop_r.mean(2)  # (B,O,D): entity axis gone

        # the target is an entity property, shared by all options -> expand over O
        if self.variant == "paired":
            target = self.metric(goal - cur)[:, None]  # (B,1,E,D)
        elif self.variant == "role_slots":
            slots = self.hyperedges(cur_r, goal_r, prop_r)
            target = self.metric(slots[:, :, 1] - slots[:, :, 0])[:, None]
        elif self.variant == "mean_hub":
            hub = self.hyperedges(cur_r, goal_r, prop_r)  # (B,E,D)
            target = self.metric(hub[:, None].expand(B, O, E, D))
        elif self.variant == "global_role_pool":
            ctx = torch.cat([cur_r.mean(1), goal_r.mean(1)], -1)
            target = self.pool(ctx)[:, None, None].expand(B, O, E, D)
        else:  # bag: one global mixed pool
            pool = torch.stack([cur_r.mean(1), goal_r.mean(1)], 0).mean(0)
            target = self.pool(torch.cat([pool, prop_pool.mean(1)], -1))[
                :, None, None
            ].expand(B, O, E, D)

        if self.variant == "slots_pooled":
            slots = self.hyperedges(cur_r, goal_r, prop_r)
            ctx = torch.cat([slots.mean((1, 2)), prop_pool.mean(1)], -1)
            target = self.pool(ctx)[:, None, None].expand(B, O, E, D)

        # the shared learned metric: -||W(a) - W(b)||^2, then pool over entities
        a = self.metric(prop_r)  # (B,O,E,D)
        dist = ((a - target) ** 2).sum(-1)  # (B,O,E)
        return -dist.min(-1).values  # (B,O): a match is when the SAME entity pairs


def oracle_accuracy(batch: int = 256) -> float:
    """Hand-coded pairing solution: match each option's proposal against the
    required change of the entity it targets, then pool over entities."""
    cur, goal, prop, label = make_batch(batch, seed=1234)
    delta = goal - cur  # (B,E,D)
    err = ((prop[:, :, :, None, :] - delta[:, None, :, None, :]) ** 2).sum(-1).sum(-1)
    score = err.min(-1).values  # (B,O): best-matching entity per option
    return float((score.argmin(-1) == label).float().mean())


def run(variant: str, steps: int = 1200, batch: int = 128, seed: int = 0):
    torch.manual_seed(seed)
    model = Toy(variant)
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    for _ in range(steps):
        cur, goal, prop, label = make_batch(batch, seed=None)
        loss = nn.functional.cross_entropy(model(cur, goal, prop), label)
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    accs = []
    with torch.inference_mode():
        for _ in range(8):  # held-out batches
            cur, goal, prop, label = make_batch(batch)
            pred = model(cur, goal, prop).argmax(-1)
            accs.append(float((pred == label).float().mean()))
    return sum(accs) / len(accs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    variants = [
        "bag",
        "global_role_pool",
        "slots_pooled",
        "mean_hub",
        "role_slots",
        "paired",
    ]
    print(
        f"task: pick the option whose proposal matches its own target entity "
        f"| chance = {1 / O:.3f}"
    )
    print(f"hand-coded pairing oracle: {oracle_accuracy():.3f}\n")
    print(f"{'variant':20s} {'accuracy (3 seeds)':>20s}   what it has")
    notes = {
        "bag": "no identity, no roles",
        "global_role_pool": "roles, no identity",
        "slots_pooled": "identity but pooled away at consumer",
        "mean_hub": "identity, pooled hub: ordering lost",
        "role_slots": "identity + ordering in slots",
        "paired": "identity + ordering, explicit pairing",
    }
    for v in variants:
        accs = [run(v, steps=args.steps, seed=s) for s in range(args.seeds)]
        print(f"{v:20s} {sum(accs) / len(accs):>20.3f}   {notes[v]}")


if __name__ == "__main__":
    main()
