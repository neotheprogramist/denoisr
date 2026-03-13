import chess
import pytest
import torch

from denoisr_chess.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr_chess.inference.engine import ChessEngine
from denoisr_chess.nn.encoder import ChessEncoder
from denoisr_chess.nn.policy_backbone import ChessPolicyBackbone
from denoisr_chess.nn.policy_head import ChessPolicyHead
from denoisr_chess.nn.value_head import ChessValueHead

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessEngine:
    @pytest.fixture
    def engine(self, device: torch.device) -> ChessEngine:
        return ChessEngine(
            encoder=ChessEncoder(122, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S,
                SMALL_NUM_HEADS,
                SMALL_NUM_LAYERS,
                SMALL_FFN_DIM,
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            board_encoder=ExtendedBoardEncoder(),
            device=device,
        )

    def test_select_move_returns_legal_move(self, engine: ChessEngine) -> None:
        board = chess.Board()
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_select_move_various_positions(self, engine: ChessEngine) -> None:
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "g1f3"):
            board.push_uci(uci)
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_evaluate_returns_wdl(self, engine: ChessEngine) -> None:
        board = chess.Board()
        wdl = engine.evaluate(board)
        assert len(wdl) == 3
        assert abs(sum(wdl) - 1.0) < 1e-5
