import sys
from pathlib import Path

import chess
import pytest

from denoisr.engine.types import EngineConfig, TimeControl
from denoisr.engine.uci_engine import UCIEngine

MOCK_ENGINE = str(Path(__file__).parent / "mock_engine.py")


def _mock_config(name: str = "MockEngine") -> EngineConfig:
    return EngineConfig(
        command=sys.executable,
        args=(MOCK_ENGINE,),
        name=name,
    )


class TestUCIEngine:
    def test_start_and_quit(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        assert engine.is_alive()
        engine.quit()
        assert not engine.is_alive()

    def test_go_returns_legal_move(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=[])
        move_uci = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000,
            btime_ms=10000,
        )
        board = chess.Board()
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
        engine.quit()

    def test_set_position_with_moves(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=["e2e4", "e7e5"])
        move_uci = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000,
            btime_ms=10000,
        )
        board = chess.Board()
        board.push_uci("e2e4")
        board.push_uci("e7e5")
        move = chess.Move.from_uci(move_uci)
        assert move in board.legal_moves
        engine.quit()

    def test_context_manager(self) -> None:
        config = _mock_config()
        with UCIEngine(config) as engine:
            engine.start()
            assert engine.is_alive()
        assert not engine.is_alive()

    def test_timeout_raises(self) -> None:
        config = EngineConfig(
            command=sys.executable,
            args=("-c", "import time; time.sleep(60)"),
            name="SlowEngine",
        )
        engine = UCIEngine(config)
        with pytest.raises(TimeoutError):
            engine.start(timeout=0.5)
        engine.quit()


class TestUCIEngineOptions:
    def test_set_option_sends_command(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_option("UCI_LimitStrength", "true")
        engine.set_option("UCI_Elo", "1200")
        assert engine.is_alive()
        engine.quit()

    def test_new_game_resets_state(self) -> None:
        engine = UCIEngine(_mock_config())
        engine.start()
        engine.set_position(fen=None, moves=["e2e4"])
        engine.new_game()
        engine.set_position(fen=None, moves=[])
        move = engine.go(
            time_control=TimeControl(base_seconds=10.0, increment=0.1),
            wtime_ms=10000,
            btime_ms=10000,
        )
        board = chess.Board()
        assert chess.Move.from_uci(move) in board.legal_moves
        engine.quit()
