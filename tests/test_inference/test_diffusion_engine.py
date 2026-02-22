import chess
import pytest
import torch

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.inference.diffusion_engine import DiffusionChessEngine
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)


class TestDiffusionChessEngine:
    @pytest.fixture
    def engine(self, device: torch.device) -> DiffusionChessEngine:
        return DiffusionChessEngine(
            encoder=ChessEncoder(122, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            diffusion=ChessDiffusionModule(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_NUM_TIMESTEPS
            ).to(device),
            schedule=CosineNoiseSchedule(SMALL_NUM_TIMESTEPS),
            board_encoder=ExtendedBoardEncoder(),
            device=device,
            num_denoising_steps=5,
        )

    def test_select_move_returns_legal(
        self, engine: DiffusionChessEngine
    ) -> None:
        board = chess.Board()
        move = engine.select_move(board)
        assert move in board.legal_moves

    def test_evaluate_returns_wdl(
        self, engine: DiffusionChessEngine
    ) -> None:
        board = chess.Board()
        wdl = engine.evaluate(board)
        assert len(wdl) == 3
        assert abs(sum(wdl) - 1.0) < 1e-5

    def test_anytime_property(self, device: torch.device) -> None:
        """Different denoising step counts should both produce legal moves."""
        args = dict(
            encoder=ChessEncoder(122, SMALL_D_S).to(device),
            backbone=ChessPolicyBackbone(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
            ).to(device),
            policy_head=ChessPolicyHead(SMALL_D_S).to(device),
            value_head=ChessValueHead(SMALL_D_S).to(device),
            diffusion=ChessDiffusionModule(
                SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_NUM_TIMESTEPS
            ).to(device),
            schedule=CosineNoiseSchedule(SMALL_NUM_TIMESTEPS),
            board_encoder=ExtendedBoardEncoder(),
            device=device,
        )
        engine_1 = DiffusionChessEngine(**args, num_denoising_steps=1)
        engine_10 = DiffusionChessEngine(**args, num_denoising_steps=10)
        board = chess.Board()
        assert engine_1.select_move(board) in board.legal_moves
        assert engine_10.select_move(board) in board.legal_moves
