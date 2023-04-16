from enum import Enum

class MaskState(Enum):
    NOT_VALID_MASK = 1
    VALID_MASK = 2
    FIRST_VALID_MASK = 3
    MATCH_EXIST = 4
    NO_MATCHES = 5

class CSOState(Enum):
    ABANDONED_OBJECT = 1
    STOLEN_OBJECT = 2
    GHOST_REGION = 3
    CSOS_NOT_VALID = 4
    CSOS_VALID = 5
    BREAK = 6