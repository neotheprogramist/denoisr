import sys
from pathlib import Path

from denoisr_chess.engine.types import TimeControl
from denoisr_chess.evaluation.benchmark import (
    BenchmarkConfig,
    BenchmarkResult,
    CompletedGame,
    run_benchmark,
)

MOCK_ENGINE = str(Path(__file__).parents[1] / "test_engine" / "mock_engine.py")


def _mock_cmd() -> str:
    return sys.executable


def _mock_args() -> tuple[str, ...]:
    return (MOCK_ENGINE,)


class TestRunBenchmark:
    def test_completes_fixed_games(self) -> None:
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=4,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            concurrency=2,
        )
        result = run_benchmark(config)
        assert isinstance(result, BenchmarkResult)
        assert result.games_played == 4
        assert result.wins + result.draws + result.losses == 4

    def test_completed_games_populated(self) -> None:
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=4,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            concurrency=2,
        )
        result = run_benchmark(config)
        assert len(result.completed_games) == 4
        for game in result.completed_games:
            assert isinstance(game, CompletedGame)
            assert game.result in {"1-0", "0-1", "1/2-1/2", "*"}
            assert game.engine1_color in {"white", "black"}
            assert isinstance(game.moves, tuple)
            assert isinstance(game.reason, str)
        # Games should be sorted by game_num
        nums = [g.game_num for g in result.completed_games]
        assert nums == sorted(nums)

    def test_sprt_can_stop_early(self) -> None:
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=1000,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            sprt_elo0=0.0,
            sprt_elo1=400.0,
            concurrency=2,
        )
        result = run_benchmark(config)
        # With identical mock engines (always draws/same results), SPRT should conclude
        assert result.games_played < 1000
        assert result.sprt_result in {"H0", "H1", None}

    def test_openings_are_used(self, tmp_path: Path) -> None:
        epd = tmp_path / "test.epd"
        epd.write_text("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1\n")
        config = BenchmarkConfig(
            engine_cmd=_mock_cmd(),
            engine_args=_mock_args(),
            opponent_cmd=_mock_cmd(),
            opponent_args=_mock_args(),
            games=2,
            time_control=TimeControl(base_seconds=60.0, increment=0.0),
            openings_path=epd,
            concurrency=1,
        )
        result = run_benchmark(config)
        assert result.games_played == 2
