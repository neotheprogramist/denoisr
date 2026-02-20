# tests/test_gui/test_match_engine.py
import sys
from pathlib import Path

import chess

from denoisr.gui.match_engine import play_game, run_match
from denoisr.gui.types import EngineConfig, MatchConfig, TimeControl
from denoisr.gui.uci_engine import UCIEngine

MOCK_ENGINE = str(Path(__file__).parent / "mock_engine.py")


def _mock_config(name: str = "MockEngine") -> EngineConfig:
    return EngineConfig(
        command=sys.executable,
        args=(MOCK_ENGINE,),
        name=name,
    )


class TestPlayGame:
    def test_game_completes(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        e1_config = _mock_config("White")
        e2_config = _mock_config("Black")
        with UCIEngine(e1_config) as white, UCIEngine(e2_config) as black:
            white.start()
            black.start()
            result = play_game(
                white=white,
                black=black,
                time_control=tc,
                max_moves=200,
            )
        assert result.result in {"1-0", "0-1", "1/2-1/2"}
        assert len(result.moves) > 0
        assert result.engine1_color == "white"

    def test_on_move_callback_called(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        e1_config = _mock_config("White")
        e2_config = _mock_config("Black")
        move_count = 0

        def on_move(board: chess.Board, uci: str) -> None:
            nonlocal move_count
            move_count += 1

        with UCIEngine(e1_config) as white, UCIEngine(e2_config) as black:
            white.start()
            black.start()
            play_game(
                white=white,
                black=black,
                time_control=tc,
                max_moves=200,
                on_move=on_move,
            )
        assert move_count > 0


class TestRunMatch:
    def test_match_completes(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        config = MatchConfig(
            engine1=_mock_config("Engine1"),
            engine2=_mock_config("Engine2"),
            games=2,
            time_control=tc,
        )
        results = run_match(config, max_moves_per_game=200)
        assert len(results) == 2
        for r in results:
            assert r.result in {"1-0", "0-1", "1/2-1/2"}

    def test_engines_alternate_colors(self) -> None:
        tc = TimeControl(base_seconds=60.0, increment=0.0)
        config = MatchConfig(
            engine1=_mock_config("Engine1"),
            engine2=_mock_config("Engine2"),
            games=2,
            time_control=tc,
        )
        results = run_match(config, max_moves_per_game=200)
        colors = [r.engine1_color for r in results]
        assert colors == ["white", "black"]
