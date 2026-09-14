from enum import Enum


class OPGate(Enum):
    NONE = -1
    OPEN = 0
    CLOSE = 1


class ENMode(Enum):
    GOAL = "Goal"
    START = "Start"
    SAMPLE = "Sample"
    SUBGOAL = "Subgoal"


class ENRole(Enum):
    NONE = -1
    START = 0  # current-scene row of one entity
    GOAL = 1  # goal row of one entity
    PRE = 2  # current value seen through a skill precondition
    POST = 3  # target/post value seen through a skill postcondition

    @staticmethod
    def size() -> int:
        return 4
