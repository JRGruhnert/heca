import torch

from heca.graphs.roles import ENRole
from heca.heca_gnn.modules.hyperedge import TypedHyperedgeLayer

DIM = 16

ROLES = torch.tensor(
    [
        ENRole.GOAL.value,
        ENRole.GOAL.value,
        ENRole.POST.value,
        ENRole.POST.value,
        ENRole.POST.value,
        ENRole.PRE.value,
        ENRole.START.value,
        ENRole.GOAL.value,
        ENRole.GOAL.value,
        ENRole.POST.value,
        ENRole.POST.value,
        ENRole.POST.value,
    ]
)
ENTITY_IDS = torch.tensor([0, 1, 0, 1, 2, 0, 0, 3, 4, 0, 0, 0])
GOAL_E0, GOAL_E1, GOAL_E3, GOAL_E4 = 0, 1, 7, 8
E0_ROWS = [2, 5, 6, 9, 10, 11]
E1_ROWS = [3]
E2_ROWS = [4]


def make_layer() -> TypedHyperedgeLayer:
    torch.manual_seed(0)
    return TypedHyperedgeLayer(DIM, num_roles=ENRole.size()).eval()


def make_rows() -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(len(ROLES), DIM)


def run(x: torch.Tensor, layer: TypedHyperedgeLayer) -> torch.Tensor:
    with torch.no_grad():
        return layer(x, ROLES, ENTITY_IDS)


def one_entity_fixture(n_members: int):
    """entity 0 with its goal row first, then ``n_members`` postcondition rows."""
    torch.manual_seed(7)
    rows = [ENRole.GOAL.value] + [ENRole.POST.value] * n_members
    return (
        torch.randn(1 + n_members, DIM),
        torch.tensor(rows),
        torch.zeros(1 + n_members, dtype=torch.long),
    )


def test_rows_are_conditioned_on_their_own_entity_goal():
    update = run(make_rows(), make_layer())
    written = update.abs().sum(dim=1) > 0
    assert bool(written[E0_ROWS + E1_ROWS].all()), "an entity with a goal got nothing"
    assert bool(
        written[[GOAL_E0, GOAL_E1, GOAL_E3, GOAL_E4]].sum() == 0
    ), "a goal row conditioned itself"
    assert bool(written[E2_ROWS].sum() == 0), "entity 2 has no goal row"


def test_every_row_of_an_entity_gets_the_goal_vector():
    update = run(make_rows(), make_layer())
    first = update[E0_ROWS[0]]
    for row in E0_ROWS[1:]:
        assert torch.allclose(update[row], first), "rows of one entity drifted apart"


def test_a_single_member_hyperedge_is_still_goal_conditioned():
    """The degeneracy the attention version had: one member, flat softmax."""
    x, roles, ids = one_entity_fixture(1)
    layer = make_layer()
    with torch.no_grad():
        before = layer(x, roles, ids)
        moved = x.clone()
        moved[0] += 5.0  # the goal row
        after = layer(moved, roles, ids)
    assert float(before[1].abs().sum()) > 0, "nothing was written"
    assert not torch.allclose(after[1], before[1]), "the goal did not reach the member"


def test_a_member_row_cannot_change_the_update():
    """Members are recipients, so their features must not enter the value."""
    x, roles, ids = one_entity_fixture(4)
    layer = make_layer()
    with torch.no_grad():
        before = layer(x, roles, ids)
        moved = x.clone()
        moved[3] += 5.0  # a member row
        after = layer(moved, roles, ids)
    assert torch.equal(after, before), "a member row leaked into the update"


def test_the_update_does_not_depend_on_the_number_of_members():
    """One member and four members must give the same update for that entity."""
    layer = make_layer()
    with torch.no_grad():
        single = layer(*one_entity_fixture(1))[1]
        many = layer(*one_entity_fixture(4))[1:]
    for row in many:
        assert torch.allclose(
            row, single
        ), f"the update changed with the member count: {row[:3]} vs {single[:3]}"


def test_a_goal_without_members_writes_nothing_and_stays_finite():
    x, roles, ids = one_entity_fixture(2)
    roles = torch.cat([roles, torch.tensor([ENRole.GOAL.value])])
    ids = torch.cat([ids, torch.tensor([1])])  # entity 1 has a goal and no rows
    x = torch.cat([x, torch.randn(1, DIM)])
    with torch.no_grad():
        update = make_layer()(x, roles, ids)
    assert torch.isfinite(update).all()
    assert torch.equal(update[-1], torch.zeros(DIM))


def test_a_foreign_entity_is_never_written():
    """The regression: e0's goal must not move e1's rows."""
    x = make_rows()
    layer = make_layer()
    before = run(x, layer)
    moved = x.clone()
    moved[GOAL_E0] += 5.0
    after = run(moved, layer)

    assert not torch.allclose(
        after[E0_ROWS[0]], before[E0_ROWS[0]]
    ), "own entity ignored"
    for row in E1_ROWS:
        assert torch.equal(after[row], before[row]), "a foreign goal leaked in"


def test_role_embedding_and_projection_receive_gradient():
    torch.manual_seed(0)
    layer = TypedHyperedgeLayer(DIM, num_roles=ENRole.size())
    out = layer(make_rows(), ROLES, ENTITY_IDS)
    out[E0_ROWS + E1_ROWS].sum().backward()

    params = dict(layer.named_parameters())
    for name in ("out.weight", "out.bias", "role_emb.weight"):
        grad = params[name].grad
        assert grad is not None, f"{name} got no gradient"
        assert bool(grad.abs().sum() > 0), f"{name} got a zero gradient"
