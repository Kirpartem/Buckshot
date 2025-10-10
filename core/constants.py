from enum import Enum
from dataclasses import dataclass

# --- Turn ---
class Turn(Enum):
    PLAYER = "player"
    DEALER = "dealer"

TURNS: list[str] = [t.value for t in Turn]

# --- Items ---
class Item(Enum):
    GLASS = "Glass"
    CIGARETTES = "Cigarettes"
    HANDCUFFS = "Handcuffs"
    SAW = "Saw"
    BEER = "Beer"

ITEMS: list[Item] = list(Item)

# --- Actions ---
class GameAction(Enum):
    SHOOT_SELF = "shoot_self"
    SHOOT_TARGET = "shoot_target"
    USE_GLASS = "use_glass"
    USE_CIGARETTES = "use_cigarettes"
    USE_HANDCUFFS = "use_handcuffs"
    USE_SAW = "use_saw"
    USE_BEER = "use_beer"

# Derived constants
GAME_ACTIONS: list[GameAction] = list(GameAction)   # full action set
ACTION_MAP = {i: action for i, action in enumerate(GameAction)}  # int → enum
ACTION_MAP_INV = {action: i for i, action in enumerate(GameAction)}  # enum → int

# --- Env constants ---
OBS_SIZE = 19
MAX_HP = 10
MAX_ITEM_COUNT = 8
MAX_CYLINDER = 6
HANDCUFF_MAX = 2

# --- StepResult ---
@dataclass
class StepResult:
    valid: bool
    action: GameAction
    prev_bot_hp: int
    prev_target_hp: int
    new_bot_hp: int
    new_target_hp: int
    player_dead: bool
    dealer_dead: bool
    terminated: bool
    info: dict



@dataclass
class SubroundCombo:
    num_items: int
    starting_hp: int
    blanks: int
    lives: int
