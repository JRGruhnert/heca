"""A hypergraph layer from scratch, with the maths next to the code.

Data layout (dense incidence; fine while n_nodes x n_hyperedges stays small):

    x          (n, d)     node features
    H          (n, m)     incidence: H[i, e] = 1 if node i is a member of hyperedge e
    node_type  (n,)       int id, small shared vocabulary
    role       (n, m)     int id: what node i does INSIDE hyperedge e (-1 for non-members)
    edge_class (m,)       int id, small shared vocabulary

The whole layer is two masked reductions plus two condition-free-per-type MLPs:

    vertex -> hyperedge:   E_e = mean_{i in e} up([x_i + type_emb, role_emb(r_ie)]) + class_emb(e)
    hyperedge -> vertex:   msg_i = mean_{e ni i} down([E_e, role_emb(r_ie)])
    update:                x_i = LayerNorm(x_i + msg_i)

Written as matrices, the *linear* form of the two reductions is just

    E = H^T X        and        X' = H E
    (optionally normalised: D_v^-1/2 H W D_e^-1 H^T D_v^-1/2 X)

so a hyperedge is: pool your members, then everyone reads their hyperedges back.
Node types / hyperedge classes / incidence roles are not structure here - they
are *inputs* to the two MLPs (embeddings added to the features, or separate
modules per id if you prefer).
"""

import torch
from torch import nn


class HypergraphLayer(nn.Module):
    def __init__(self, d: int, n_node_types: int, n_roles: int, n_edge_classes: int):
        super().__init__()
        # the three differentiation mechanisms, all just embedding tables
        self.node_type = nn.Embedding(n_node_types, d)
        self.role = nn.Embedding(n_roles, d)
        self.edge_class = nn.Embedding(n_edge_classes, d)
        # two shared kernels; a per-type variant would be a ModuleDict here
        self.up = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.down = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        self.norm = nn.LayerNorm(d)

    def forward(self, x, H, node_type, role, edge_class):
        n, m = H.shape
        member = H > 0  # (n, m) bool

        # ---- vertex -> hyperedge -------------------------------------------
        # every (node, hyperedge) incidence gets its own message, conditioned on
        # the role that node plays in that hyperedge
        h_node = x + self.node_type(node_type)  # (n, d)
        member_feat = h_node[:, None, :].expand(n, m, -1)  # (n, m, d)
        role_feat = self.role(role.clamp(min=0))  # (n, m, d)
        up_msg = self.up(torch.cat([member_feat, role_feat], -1))  # (n, m, d)

        up_msg = up_msg * member[..., None]  # ignore non-members
        # sum over NODES (dim 0) -> one state per hyperedge
        E = up_msg.sum(0) / member.sum(0).unsqueeze(-1).clamp(min=1)  # (m, d)
        E = E + self.edge_class(edge_class)  # hyperedge class is added here

        # ---- hyperedge -> vertex -------------------------------------------
        e_feat = E[None, :, :].expand(n, m, -1)  # (n, m, d)
        down_msg = self.down(torch.cat([e_feat, role_feat], -1))  # (n, m, d)
        down_msg = down_msg * member[..., None]
        msg = down_msg.sum(1) / member.sum(1, keepdim=True).clamp(min=1)  # (n, d)

        # ---- residual (the node keeps its own value) ------------------------
        return self.norm(x + msg), E

    def forward_sparse(self, x, edge_index, role, edge_class, node_type, n_hyperedges):
        """Same thing as libraries do it: index_add instead of a dense mask.

        edge_index[0] = member node, edge_index[1] = hyperedge id.
        """
        src, dst = edge_index
        up_msg = self.up(
            torch.cat([x[src] + self.node_type(node_type[src]), self.role(role)], -1)
        )
        E = torch.zeros(n_hyperedges, x.shape[1], dtype=x.dtype)
        E.index_add_(0, dst, up_msg)  # sum members into their hyperedge
        count = torch.zeros(n_hyperedges, 1).index_add_(0, dst, torch.ones(len(dst), 1))
        E = E / count.clamp(min=1) + self.edge_class(edge_class)
        down_msg = self.down(torch.cat([E[dst], self.role(role)], -1))
        # back to members, averaged over the hyperedges each node belongs to
        msg = torch.zeros_like(x).index_add_(0, src, down_msg)
        count_v = torch.zeros(x.shape[0], 1).index_add_(0, src, torch.ones(len(src), 1))
        msg = msg / count_v.clamp(min=1)
        return self.norm(x + msg), E


def main():
    torch.manual_seed(0)
    n, m, d = 6, 3, 8
    # two entities (faucet0 rows 0-2, cube0 rows 3-5) as three hyperedges:
    #   e0 = "entity faucet0", e1 = "entity cube0", e2 = "option close_faucet"
    H = torch.tensor(
        [
            [1, 0, 1],  # faucet0.cur   is in faucet0-group and in the option
            [1, 0, 1],  # faucet0.goal  is in faucet0-group and in the option
            [1, 0, 1],  # faucet0.post  is in faucet0-group and in the option
            [0, 1, 0],  # cube0.cur
            [0, 1, 0],  # cube0.goal
            [0, 1, 0],  # cube0.post
        ]
    )
    node_type = torch.tensor([2, 2, 2, 0, 0, 0])  # 2 = prismatic, 0 = free
    edge_class = torch.tensor([0, 0, 1])  # 0 = entity group, 1 = option context
    role = torch.full((n, m), -1)
    role[:3, 0] = torch.tensor([0, 1, 2])  # cur / goal / post inside faucet0-group
    role[3:, 1] = torch.tensor([0, 1, 2])
    role[:3, 2] = torch.tensor([0, 1, 2])  # same rows, option context

    layer = HypergraphLayer(d, n_node_types=4, n_roles=3, n_edge_classes=2)
    x = torch.randn(n, d)
    x2, E = layer(x, H, node_type, role, edge_class)
    print(
        f"x {tuple(x.shape)} -> x' {tuple(x2.shape)} | hyperedge states E {tuple(E.shape)}"
    )
    print(f"mean |change| {float((x2 - x).abs().mean()):.4f}")

    # 1. what a hyperedge DOES: members exchange information. Perturb one member
    #    of the faucet0 group and see another member of the same group change.
    x_pert = x.clone()
    x_pert[2] += 5.0  # faucet0.post
    out_pert, _ = layer(x_pert, H, node_type, role, edge_class)
    print(
        "\nsame hyperedge, other member (row 0, faucet0.cur) changed by "
        f"{float((out_pert[0] - x2[0]).abs().mean()):.4f}"
    )
    # ... while a node in a different hyperedge is unaffected
    print(
        "different hyperedge (row 3, cube0.cur) changed by "
        f"{float((out_pert[3] - x2[3]).abs().mean()):.6f}"
    )

    # 2. the three differentiation mechanisms, one at a time
    role2 = role.clone()
    role2[0, 2] = 1  # row 0 now plays "goal" instead of "cur" in the option hyperedge
    out_role, _ = layer(x, H, node_type, role2, edge_class)
    print(
        f"\nchange ROLE of one incidence: row 0 changes by "
        f"{float((out_role[0] - x2[0]).abs().mean()):.4f} "
        f"(row 1, a co-member, changes by {float((out_role[1] - x2[1]).abs().mean()):.4f})"
    )
    types2 = node_type.clone()
    types2[0] = 0
    out_type, _ = layer(x, H, types2, role, edge_class)
    print(
        f"change NODE TYPE of row 0: row 0 changes by "
        f"{float((out_type[0] - x2[0]).abs().mean()):.4f}"
    )
    classes2 = edge_class.clone()
    classes2[0] = 1
    out_class, _ = layer(x, H, node_type, role, classes2)
    print(
        f"change HYPEREDGE CLASS of e0: row 0 changes by "
        f"{float((out_class[0] - x2[0]).abs().mean()):.4f}"
    )

    # 3. the same computation in sparse (library-style) form
    edge_index = H.nonzero().t()
    xs, Es = layer.forward_sparse(
        x,
        edge_index,
        role[edge_index[0], edge_index[1]],
        edge_class,
        node_type,
        m,
    )
    print(
        f"\nsparse form agrees with dense: "
        f"{bool(torch.allclose(xs, x2, atol=1e-5))}"
    )


if __name__ == "__main__":
    main()
