"""Tests for ACPL analyzer."""

import shutil

import pytest

from denoisr_chess.evaluation.analyzer import (
    ACPL_ELO_INTERCEPT,
    ELO_MAX,
    ELO_MIN,
    acpl_to_elo,
    run_analysis,
)
from denoisr_chess.evaluation.benchmark import CompletedGame


class TestAcplToElo:
    def test_zero_acpl_gives_max_elo(self) -> None:
        assert acpl_to_elo(0.0) == ELO_MAX

    def test_high_acpl_gives_min_elo(self) -> None:
        assert acpl_to_elo(200.0) == ELO_MIN

    def test_typical_gm_acpl(self) -> None:
        elo = acpl_to_elo(15.0)
        assert 2700 < elo < 3100

    def test_typical_beginner_acpl(self) -> None:
        elo = acpl_to_elo(80.0)
        assert 600 < elo < 800

    def test_clamp_low(self) -> None:
        assert acpl_to_elo(1000.0) == ELO_MIN

    def test_clamp_high(self) -> None:
        assert acpl_to_elo(-10.0) == ELO_MAX

    def test_linearity(self) -> None:
        elo_a = acpl_to_elo(10.0)
        elo_b = acpl_to_elo(20.0)
        # 10 ACPL difference should be ~330 Elo
        assert abs((elo_a - elo_b) - 330.0) < 1.0

    def test_exact_midpoint(self) -> None:
        elo = acpl_to_elo(50.0)
        assert abs(elo - (ACPL_ELO_INTERCEPT - 33 * 50)) < 0.01


class TestRunAnalysis:
    @pytest.mark.skipif(
        shutil.which("stockfish") is None,
        reason="Stockfish not installed",
    )
    def test_basic_analysis(self) -> None:
        """Integration test — requires Stockfish on PATH."""
        game = CompletedGame(
            game_num=0,
            result="1-0",
            engine1_color="white",
            moves=("e2e4", "e7e5", "f1c4", "b8c6", "d1h5", "g8f6", "h5f7"),
            start_fen=None,
            reason="checkmate",
        )
        result = run_analysis([game], depth=8, concurrency=1)
        assert len(result.game_analyses) == 1
        assert result.overall_acpl >= 0.0
        assert result.estimated_elo >= ELO_MIN
        assert result.estimated_elo <= ELO_MAX

    def test_empty_games(self) -> None:
        result = run_analysis([], depth=8, concurrency=1)
        assert len(result.game_analyses) == 0
        assert result.overall_acpl == 0.0

    def test_no_move_games_skipped(self) -> None:
        game = CompletedGame(
            game_num=0,
            result="*",
            engine1_color="white",
            moves=(),
            start_fen=None,
            reason="stopped",
        )
        result = run_analysis([game], depth=8, concurrency=1)
        assert len(result.game_analyses) == 0
