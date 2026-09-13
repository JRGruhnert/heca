ROLE_OTHER = 0  # e.g. fitted mixture components
ROLE_CURRENT = 1  # current-scene row of one entity
ROLE_GOAL = 2  # goal row of one entity
ROLE_PRE = 3  # current value seen through a skill precondition
ROLE_POST = 4  # target/post value seen through a skill postcondition

# Option-state slots: which options an edge into the slot comes from. The slots
# are the three nodes of ``OptionStateNodes``, and the edge attribute is
# ``1 - role``: +1 ungated, 0 all, -1 gated.
ROLE_UNGATED = 0
ROLE_ALL = 1
ROLE_GATED = 2
