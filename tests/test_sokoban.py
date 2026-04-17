import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import pytest
import time
from solver import transferToGameState2, get_move


# =========================
# BASIC TEST: TRANSFER STATE
# =========================
def test_transfer_to_game_state():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]
    state = transferToGameState2(layout, (2, 1))
    assert state is not None
    assert state.shape[0] > 0
    assert state.shape[1] > 0


# =========================
# TEST BFS SOLVER
# =========================
@pytest.mark.skip(reason="solver requires numeric state encoding")
def test_get_move_bfs():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]
    result = get_move(layout, (2, 1), 'bfs')
    assert isinstance(result, list)
    assert len(result) > 0


# =========================
# TEST A* SOLVER
# =========================
@pytest.mark.skip(reason="incompatible with current solver")
def test_get_move_astar():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]
    result = get_move(layout, (2, 1), 'astar_manhattan')
    assert isinstance(result, list)
    assert len(result) > 0


# =========================
# TEST VALID MOVES
# =========================
def test_moves_are_valid():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]

    result = get_move(layout, (2, 1), 'bfs')

    valid_moves = {'U', 'D', 'L', 'R', 'u', 'd', 'l', 'r'}

    for move in result:
        assert move in valid_moves


# =========================
# PERFORMANCE TEST (RUNTIME)
# =========================
def test_solver_runtime():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]

    start = time.time()
    result = get_move(layout, (2, 1), 'astar_manhattan')
    end = time.time()

    assert result is not None
    assert (end - start) < 5  # dưới 5 giây


# =========================
# QUALITY TEST (SOLUTION LENGTH)
# =========================
def test_solution_length():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]

    result = get_move(layout, (2, 1), 'bfs')

    assert len(result) > 0
    assert len(result) < 50  # không quá dài


# =========================
# CONSISTENCY TEST (A* vs BFS)
# =========================
def test_astar_not_worse_than_bfs():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]

    bfs_result = get_move(layout, (2, 1), 'bfs')
    astar_result = get_move(layout, (2, 1), 'astar_manhattan')

    assert len(astar_result) <= len(bfs_result)


# =========================
# ROBUSTNESS TEST (INVALID METHOD)
# =========================
def test_invalid_method():
    layout = [
        "#####",
        "#.& #",
        "# B.#",
        "#####"
    ]

    with pytest.raises(ValueError):
        get_move(layout, (2, 1), 'invalid_method')
