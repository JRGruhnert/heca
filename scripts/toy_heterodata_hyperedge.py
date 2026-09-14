"""How to put a hypergraph into a PyG HeteroData object.

`HeteroData` has no hyperedge primitive, so a hyperedge becomes **a node type**,
and the incidence becomes **two edge types** (member -> group, group -> member).
Per-incidence roles and per-hyperedge classes are then just extra attributes on
those edges / nodes. Verified on torch_geometric 2.5.0.

Two routes:

A. explicit (this file's `build_hetero`): hyperedges as nodes. Full control, and
   you can condition the up/down messages on the incidence role in both
   directions.
B. native: `HyperGraphData` + `HypergraphConv`, which stores the incidence in one
   `hyperedge_index` tensor of shape (2, nnz) - row 0 = node, row 1 = hyperedge id
   - and takes per-incidence `hyperedge_weight` and per-hyperedge
   `hyperedge_attr`. Less code, but the incidence role can only enter as a
   weight/attr, not as a full message input.
"""

import torch
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HypergraphConv, MessagePassing


def build_hetero() -> tuple[HeteroData, torch.Tensor]:
    # incidence: rows x groups. 6 rows = 2 entities x (cur, goal, post);
    # 3 groups = entity faucet0, entity cube0, option "close faucet0"
    H = torch.tensor(
        [
            [1, 0, 1],  # faucet0.cur
            [1, 0, 1],  # faucet0.goal
            [1, 0, 1],  # faucet0.post
            [0, 1, 0],  # cube0.cur
            [0, 1, 0],  # cube0.goal
            [0, 1, 0],  # cube0.post
        ]
    )
    member, group = H.nonzero().t()  # (nnz,) (nnz,) : both directions

    d = HeteroData()
    # --- the "vertices" of the hypergraph -------------------------------
    d["row"].x = torch.randn(6, 8)
    d["row"].type_id = torch.tensor([2, 2, 2, 0, 0, 0])  # shared vocabulary
    # --- the "hyperedges", materialised as a node type -------------------
    d["group"].class_id = torch.tensor([0, 0, 1])  # 0 = entity, 1 = option
    d["group"].x = torch.zeros(3, 8)  # optional own state (else purely pooled)
    # --- the incidence, as two edge types -------------------------------
    d["row", "member", "group"].edge_index = torch.stack([member, group])
    d["group", "member", "row"].edge_index = torch.stack([group, member])
    # role of a row INSIDE that group: ONE ID PER INCIDENCE (9 here), not per row
    role = member % 3  # cur / goal / post, per incidence
    d["row", "member", "group"].role_id = role
    d["group", "member", "row"].role_id = role
    assert role.numel() == member.numel()
    return d, H


class Up(MessagePassing):
    """row -> group: pool the members, conditioned on each member's role."""

    def __init__(self, dim: int, n_roles: int, n_classes: int):
        super().__init__(aggr="mean", node_dim=0)  # mean over members
        self.role = nn.Embedding(n_roles, dim)
        self.cls = nn.Embedding(n_classes, dim)  # hyperedge class, added here
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, x_row, edge_index, role_id, class_id, size):
        # size=(n_rows, n_groups): on a bipartite incidence PyG cannot infer both
        # sides from the tensors, so pass it explicitly
        out = self.propagate(edge_index, x=x_row, edge_attr=role_id, size=size)
        return out + self.cls(class_id)  # (n_groups, dim)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, self.role(edge_attr)], dim=-1))


class Down(MessagePassing):
    """group -> row: every member reads its group's state, role-conditioned."""

    def __init__(self, dim: int, n_roles: int):
        super().__init__(aggr="mean", node_dim=0)  # mean over a row's groups
        self.role = nn.Embedding(n_roles, dim)
        self.mlp = nn.Sequential(nn.Linear(2 * dim, dim), nn.GELU())

    def forward(self, x_group, edge_index, role_id, size):
        return self.propagate(edge_index, x=x_group, edge_attr=role_id, size=size)

    def message(self, x_j, edge_attr):
        return self.mlp(torch.cat([x_j, self.role(edge_attr)], dim=-1))


def main():
    data, H = build_hetero()
    print(data)
    dim, n_groups, n_rows = 8, 3, 6
    torch.manual_seed(0)
    up, down = Up(dim, 3, 2), Down(dim, 3)

    # one hyperedge round
    g = up(
        data["row"].x,
        data["row", "member", "group"].edge_index,
        data["row", "member", "group"].role_id,
        data["group"].class_id,
        size=(n_rows, n_groups),
    )
    r = down(
        g,
        data["group", "member", "row"].edge_index,
        data["group", "member", "row"].role_id,
        size=(n_groups, n_rows),
    )
    x_new = torch.nn.functional.layer_norm(data["row"].x + r, (dim,))
    print(f"\ngroup states {tuple(g.shape)}, row update {tuple(x_new.shape)}")
    print(f"mean |change| {float((x_new - data['row'].x).abs().mean()):.4f}")

    # rows that share a group exchange information; rows that do not, do not
    x2 = data["row"].x.clone()
    x2[2] += 5.0  # faucet0.post
    g2 = up(
        x2,
        data["row", "member", "group"].edge_index,
        data["row", "member", "group"].role_id,
        data["group"].class_id,
        size=(n_rows, n_groups),
    )
    r2 = down(
        g2,
        data["group", "member", "row"].edge_index,
        data["group", "member", "row"].role_id,
        size=(n_groups, n_rows),
    )
    print(f"same group (row 0) changed by {float((r2[0] - r[0]).abs().mean()):.4f}")
    print(f"other group (row 3) changed by {float((r2[3] - r[3]).abs().mean()):.6f}")

    # B: the native route - one incidence tensor, per-hyperedge attrs
    conv = HypergraphConv(dim, dim, use_attention=True)
    hyperedge_index = torch.stack([H.nonzero()[:, 0], H.nonzero()[:, 1]])
    hyperedge_weight = torch.ones(hyperedge_index.shape[1])  # per incidence
    hyperedge_attr = torch.randn(n_groups, dim)  # per hyperedge
    out = conv(data["row"].x, hyperedge_index, hyperedge_weight, hyperedge_attr)
    print(
        f"\nnative HypergraphConv output {tuple(out.shape)} from "
        f"hyperedge_index {tuple(hyperedge_index.shape)}"
    )


if __name__ == "__main__":
    main()
