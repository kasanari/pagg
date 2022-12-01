from enum import Enum


class STEP(Enum):
    AND = "and"
    OR = "or"
    DEFENSE = "defense"


class TTC(Enum):
    """Time to compromise"""

    NONE = "none"
    DEFAULT = "default"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class REWARD(Enum):
    """Reward for completing a step"""

    DEFAULT = "default"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


ASSET = "asset"
STEP_TYPE = "step_type"
CONDITIONS = "conditions"
