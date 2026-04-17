import sys
sys.path.append(".")

from level import Level
from solver import (
    get_move,
    transferToGameState,
    heuristic_manhattan,
)

# =========================
# TEST LEVEL LOADING
# =========================
def test_load_level():
    level = Level(1)

    assert level.structure is not None
    assert len(level.structure) > 0
    assert level.position_player is not None


# =========================
# TEST SOLVER (BFS)
# =========================
def test_solver_bfs_runs():
    level = Level(1)

    result = get_move(level.structure[:-1], level.position_player, 'bfs')

    assert result is not None
    assert isinstance(result, list)


# =========================
# TEST SOLVER (A*)
# =========================
def test_solver_astar_runs():
    level = Level(1)

    result = get_move(level.structure[:-1], level.position_player, 'astar_manhattan')

    assert result is not None
    assert isinstance(result, list)


# =========================
# TEST HEURISTIC
# =========================
def test_heuristic_manhattan():
    player = (1, 1)
    boxes = [(2, 2)]

    h = heuristic_manhattan(player, boxes)

    assert h >= 0


# =========================
# TEST TRANSFER STATE
# =========================
def test_transfer_state():
    layout = [
        "#####",
        "#& B#",
        "# . #",
        "#####"
    ]

    state = transferToGameState(layout)

    assert state is not None


# =========================
# TEST LEVEL STRUCTURE VALID
# =========================
def test_level_structure_valid():
    level = Level(1)

    for row in level.structure:
        for cell in row:
            assert cell is not None


# =========================
# TEST PLAYER POSITION IN BOUNDS
# =========================
def test_player_position_valid():
    level = Level(1)

    x, y = level.position_player

    assert x >= 0
    assert y >= 0
    assert y < len(level.structure)
    assert x < len(level.structure[0])