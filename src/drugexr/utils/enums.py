from enum import Enum


class RewardScheme(Enum):
    PARETO_FRONT = 1
    WEIGHTED_SUM = 2
    CROWDING_DISTANCE = 3


class MeanFn(Enum):
    GEOMETRIC = 1
    ARITHMETIC = 2
