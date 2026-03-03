import shutil
from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import chess
import chess.engine
import pytest

from denoisr.data.stockfish_oracle import StockfishOracle

STOCKFISH_PATH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(
    STOCKFISH_PATH is None, reason="stockfish not installed"
)


class TestStockfishOracle:
    @pytest.fixture
    def oracle(self) -> Iterator[StockfishOracle]:
        assert STOCKFISH_PATH is not None
        o = StockfishOracle(path=STOCKFISH_PATH, depth=10)
        yield o
        o.close()

    def test_starting_position_policy_is_distribution(
        self, oracle: StockfishOracle
    ) -> None:
        policy, _, _ = oracle.evaluate(chess.Board())
        total = policy.data.sum().item()
        assert abs(total - 1.0) < 0.01

    def test_starting_position_value_is_wdl(self, oracle: StockfishOracle) -> None:
        _, value, _ = oracle.evaluate(chess.Board())
        assert abs(value.win + value.draw + value.loss - 1.0) < 1e-5

    def test_eval_is_finite(self, oracle: StockfishOracle) -> None:
        _, _, cp = oracle.evaluate(chess.Board())
        assert -10000 <= cp <= 10000

    def test_mate_position_value(self, oracle: StockfishOracle) -> None:
        """Oracle should handle a checkmated position without crashing."""
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push_uci(uci)
        policy, value, _ = oracle.evaluate(board)
        # No legal moves → policy should be all zeros
        assert policy.data.sum().item() == pytest.approx(0.0)
        # WDL should still sum to 1
        assert abs(value.win + value.draw + value.loss - 1.0) < 1e-5

    def test_best_move_has_substantial_probability(
        self, oracle: StockfishOracle
    ) -> None:
        """Best move should be above uniform probability."""
        board = chess.Board()
        legal_count = board.legal_moves.count()
        policy, _, _ = oracle.evaluate(board)
        max_prob = policy.data.max().item()
        uniform_prob = 1.0 / legal_count
        # Best move should be above uniform baseline
        assert max_prob > uniform_prob

    def test_policy_only_on_legal_moves(self, oracle: StockfishOracle) -> None:
        board = chess.Board()
        policy, _, _ = oracle.evaluate(board)
        # Check that nonzero entries correspond to legal moves
        for from_sq in range(64):
            for to_sq in range(64):
                if policy.data[from_sq, to_sq].item() > 0:
                    found = any(
                        m.from_square == from_sq and m.to_square == to_sq
                        for m in board.legal_moves
                    )
                    assert found, (
                        f"Policy nonzero at ({from_sq},{to_sq}) but no legal move"
                    )

    def test_temperature_changes_distribution_sharpness(
        self, oracle: StockfishOracle
    ) -> None:
        """Lower temperature should produce sharper (more peaked) distributions."""
        board = chess.Board()

        assert STOCKFISH_PATH is not None
        sharp_oracle = StockfishOracle(
            path=STOCKFISH_PATH, depth=10, policy_temperature=30.0
        )
        soft_oracle = StockfishOracle(
            path=STOCKFISH_PATH, depth=10, policy_temperature=150.0
        )

        sharp_policy, _, _ = sharp_oracle.evaluate(board)
        soft_policy, _, _ = soft_oracle.evaluate(board)

        # Sharper distribution (lower temperature) should have higher max probability
        assert sharp_policy.data.max().item() > soft_policy.data.max().item()

        sharp_oracle.close()
        soft_oracle.close()

    def test_label_smoothing_redistributes_mass(self, oracle: StockfishOracle) -> None:
        """With label smoothing, minimum probability on legal moves should be > 0."""
        board = chess.Board()
        assert STOCKFISH_PATH is not None
        smoothed = StockfishOracle(
            path=STOCKFISH_PATH,
            depth=10,
            policy_temperature=150.0,
            label_smoothing=0.1,
        )
        policy, _, _ = smoothed.evaluate(board)
        # All legal moves should have nonzero probability
        legal_probs = policy.data[policy.data > 0]
        assert legal_probs.min().item() > 0.001  # smoothing ensures minimum
        smoothed.close()

    def test_missing_wdl_raises_value_error(self, oracle: StockfishOracle) -> None:
        """ValueError is raised when Stockfish returns no WDL data."""
        mock_score = MagicMock()
        mock_score.white.return_value = chess.engine.Cp(30)
        info_no_wdl: dict[str, object] = {"score": mock_score}

        with patch.object(oracle._engine, "analyse", return_value=info_no_wdl):
            with pytest.raises(ValueError, match="WDL"):
                oracle._get_value(chess.Board())
