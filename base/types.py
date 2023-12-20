from enum import IntEnum


class IneqType(IntEnum):
    LOWER_OR_EQUALS = -1
    EQUALS = 0
    GREATER_OR_EQUALS = 1
    MUTEX = 2
    BOUNDS = 3