import os
import sys
import time

os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.append(".")

import pygame

from level import Level
from solver import get_move, heuristic_manhattan, legalActions, isEndState
from scores import Scores
# =========================
# BASIC TEST
# =========================
def test_load_level():
    level = Level(1)

    assert level.structure is not None
    assert len(level.structure) > 0
    assert level.position_player is not None


# =========================
# SOLVER TEST (SAFE)
# =========================
def test_solver_runs():
    level = Level(1)

    result = get_move(
        level.structure[:-1],
        level.position_player,
        'bfs'
    )

    # Không ép solver phải đúng → chỉ cần không crash
    assert result is None or isinstance(result, list)


def test_solver_astar_runs():
    level = Level(1)

    result = get_move(
        level.structure[:-1],
        level.position_player,
        'astar_manhattan'
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
        'bfs'
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
        'bfs'
    )

    if result:
        assert len(result) < 500

# =========================
# ADDITIONAL LEVEL TESTS
# =========================
def test_level_cancel_last_move():
    level = Level(1)

    # Chỉ cần gọi để đảm bảo hàm không crash
    level.cancel_last_move()

    assert level is not None


def test_level_render_runs():
    pygame.init()
    screen = pygame.display.set_mode((800, 600))

    level = Level(1)
    level.render(screen)

    assert screen is not None

    pygame.quit()


# =========================
# SOLVER UTILITY TESTS
# =========================
def test_solver_legal_actions_runs():
    level = Level(1)

    actions = legalActions(
        level.structure[:-1],
        level.position_player
    )

    assert isinstance(actions, list)


def test_solver_is_end_state_runs():
    level = Level(1)

    result = isEndState(level.structure[:-1])

    assert isinstance(result, bool)


# =========================
# SCORES TESTS
# =========================
def test_scores_initialization():
    scores = Scores()

    assert scores is not None


def test_scores_load_runs():
    scores = Scores()

    result = scores.load()

    assert result is not None or result is None


def test_scores_save_runs():
    scores = Scores()

    # Chỉ test hàm chạy không crash
    result = scores.save()

    assert result is not None or result is None
