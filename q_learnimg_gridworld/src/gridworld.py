from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

ROWS = 4
COLS = 4
START = 0
GOAL = 15
OBSTACLES = {5, 10}

# Action IDs: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT
ACTIONS: Dict[int, Tuple[int, int]] = {
    0: (-1, 0),
    1: (1, 0),
    2: (0, -1),
    3: (0, 1),
}
ACTION_NAMES = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ARROWS = {0: "↑", 1: "↓", 2: "←", 3: "→"}
NUM_STATES = ROWS * COLS
NUM_ACTIONS = len(ACTIONS)


@dataclass(frozen=True)
class Transition:
    next_state: int
    reward: float
    done: bool


def state_to_position(state: int) -> tuple[int, int]:
    return state // COLS, state % COLS


def position_to_state(row: int, col: int) -> int:
    return row * COLS + col


def step(state: int, action: int) -> Transition:
    row, col = state_to_position(state)
    d_row, d_col = ACTIONS[action]
    new_row, new_col = row + d_row, col + d_col

    if not (0 <= new_row < ROWS and 0 <= new_col < COLS):
        return Transition(state, -1.0, False)

    next_state = position_to_state(new_row, new_col)

    if next_state in OBSTACLES:
        return Transition(state, -10.0, False)

    if next_state == GOAL:
        return Transition(next_state, 10.0, True)

    return Transition(next_state, -1.0, False)
