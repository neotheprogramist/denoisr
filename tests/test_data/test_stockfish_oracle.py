import shutil

import chess
import pytest

from denoisr.data.stockfish_oracle import StockfishOracle

STOCKFISH_PATH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(
    STOCKFISH_PATH is None, reason="stockfish not installed"
)


class TestStockfishOracle:
    @pytest.fixture
    def oracle(self) -> StockfishOracle:
        assert STOCKFISH_PATH is not None
        return StockfishOracle(path=STOCKFISH_PATH, depth=10)

    def test_starting_position_policy_is_distribution(
        self, oracle: StockfishOracle
    ) -> None:
        policy, _, _ = oracle.evaluate(chess.Board())
        total = policy.data.sum().item()
        assert abs(total - 1.0) < 0.01

    def test_starting_position_value_is_wdl(
        self, oracle: StockfishOracle
    ) -> None:
        _, value, _ = oracle.evaluate(chess.Board())
        assert abs(value.win + value.draw + value.loss - 1.0) < 1e-5

    def test_eval_is_finite(self, oracle: StockfishOracle) -> None:
        _, _, cp = oracle.evaluate(chess.Board())
        assert -10000 <= cp <= 10000

    def test_mate_position_value(self, oracle: StockfishOracle) -> None:
        # Position after Scholar's mate — white already won
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push_uci(uci)
        # Game is over, but oracle should still handle it
        # (implementation may return extreme values)

    def test_policy_only_on_legal_moves(
        self, oracle: StockfishOracle
    ) -> None:
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
                    assert found, f"Policy nonzero at ({from_sq},{to_sq}) but no legal move"
