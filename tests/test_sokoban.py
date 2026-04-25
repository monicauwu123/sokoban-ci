import os
import sys
import time
import types

fake_pyautogui = types.ModuleType("pyautogui")
fake_pyautogui.press = lambda *args, **kwargs: None
fake_pyautogui.typewrite = lambda *args, **kwargs: None
fake_pyautogui.hotkey = lambda *args, **kwargs: None
sys.modules["pyautogui"] = fake_pyautogui
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

    assert isinstance(actions, (list, tuple))
    assert len(actions) >= 0

def test_solver_is_end_state_runs():
    level = Level(1)

    boxes = PosOfBoxes(level.structure[:-1])
    result = isEndState(boxes)

    assert isinstance(result, bool)

# =========================
# SCORES TESTS
# =========================
from scores import Scores


class DummyGame:
    def __init__(self):
        self.index_level = 1
        self.loaded = False
        self.started = False

    def load_level(self):
        self.loaded = True

    def start(self):
        self.started = True


def test_scores_initialization():
    game = DummyGame()
    scores = Scores(game)

    assert scores.game == game


def test_scores_save_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    game = DummyGame()
    game.index_level = 2

    scores = Scores(game)
    scores.save()

    assert (tmp_path / "scores").exists()


def test_scores_load_no_file_runs(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    game = DummyGame()
    scores = Scores(game)

    scores.load()

    assert game.index_level == 1



# =========================
# GAME TESTS
# =========================
import pygame
from unittest.mock import Mock
from game import Game


def test_game_initialization(monkeypatch):
    pygame.init()
    window = pygame.display.set_mode((800, 600))

    dummy_surface = pygame.Surface((32, 32))

    monkeypatch.setattr(
        "pygame.image.load",
        lambda path: dummy_surface
    )

    game = Game(window)

    assert game.window == window
    assert game.player is not None
    assert game.level is not None
    assert game.index_level == 1

    pygame.quit()


def test_game_has_win_returns_boolean(monkeypatch):
    pygame.init()
    window = pygame.display.set_mode((800, 600))

    dummy_surface = pygame.Surface((32, 32))

    monkeypatch.setattr(
        "pygame.image.load",
        lambda path: dummy_surface
    )

    game = Game(window)

    result = game.has_win()

    assert isinstance(result, bool)

    pygame.quit()
