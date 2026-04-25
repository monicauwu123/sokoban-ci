import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.append(".")

from level import Level
from solver import (
    get_move,
    heuristic_manhattan,
    legalActions,
    isEndState,
    PosOfBoxes,
)


# =========================
# BASIC TEST
# =========================
def test_load_level():
    level = Level(1)

    assert level.structure is not None
    assert len(level.structure) > 0
    assert level.position_player is not None


# =========================
# SOLVER TESTS
# =========================
def test_solver_runs():
    level = Level(1)

    result = get_move(
        level.structure[:-1],
        level.position_player,
        "bfs",
    )

    assert result is None or isinstance(result, list)


def test_solver_astar_runs():
    level = Level(1)

    result = get_move(
        level.structure[:-1],
        level.position_player,
        "astar_manhattan",
    )

    assert result is None or isinstance(result, list)


# =========================
# HEURISTIC TEST
# =========================
def test_heuristic():
    h = heuristic_manhattan((1, 1), [(2, 2)])
    assert h >= 0


# =========================
# PERFORMANCE METRIC
# =========================
def test_runtime():
    level = Level(1)

    start = time.time()

    get_move(
        level.structure[:-1],
        level.position_player,
        "bfs",
    )

    end = time.time()
    runtime = end - start

    print(f"\nSolver runtime: {runtime:.4f} seconds")

    assert runtime < 10


# =========================
# QUALITY METRIC
# =========================
def test_solution_length_if_exists():
    level = Level(1)

    result = get_move(
        level.structure[:-1],
        level.position_player,
        "bfs",
    )

    if result:
        assert len(result) < 500


# =========================
# ADDITIONAL SOLVER UTILITY TESTS
# =========================
def test_solver_legal_actions_runs():
    level = Level(1)

    actions = legalActions(
        level.position_player,
        PosOfBoxes(level.structure[:-1]),
    )

    assert isinstance(actions, list)


def test_solver_is_end_state_runs():
    level = Level(1)

    boxes = PosOfBoxes(level.structure[:-1])
    result = isEndState(boxes)

    assert isinstance(result, bool)
