import chess
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.dataset import ChessDataset, generate_examples_from_game
from denoisr.types import (
    Action,
    BoardTensor,
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


def _make_example(result: float = 1.0) -> TrainingExample:
    return TrainingExample(
        board=BoardTensor(torch.zeros(12, 8, 8)),
        policy=PolicyTarget(torch.zeros(64, 64)),
        value=ValueTarget(
            win=max(result, 0.0),
            draw=1.0 - abs(result),
            loss=max(-result, 0.0),
        ),
    )


class TestChessDataset:
    def test_length(self) -> None:
        examples = [_make_example() for _ in range(10)]
        ds = ChessDataset(examples)
        assert len(ds) == 10

    def test_getitem_shapes(self) -> None:
        ds = ChessDataset([_make_example()])
        board, policy, value = ds[0]
        assert board.shape == (12, 8, 8)
        assert policy.shape == (64, 64)
        assert value.shape == (3,)

    def test_value_tensor_order(self) -> None:
        ex = _make_example(result=1.0)
        ds = ChessDataset([ex])
        _, _, value = ds[0]
        assert value[0].item() == ex.value.win
        assert value[1].item() == ex.value.draw
        assert value[2].item() == ex.value.loss


class TestGenerateExamples:
    def test_scholars_mate(self) -> None:
        board = chess.Board()
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        record = GameRecord(
            actions=tuple(
                Action(
                    chess.Move.from_uci(m).from_square,
                    chess.Move.from_uci(m).to_square,
                    chess.Move.from_uci(m).promotion,
                )
                for m in moves
            ),
            result=1.0,
        )
        encoder = SimpleBoardEncoder()
        examples = list(generate_examples_from_game(record, encoder))
        assert len(examples) == 7  # one per position before each move
        for ex in examples:
            assert ex.board.data.shape == (12, 8, 8)
            assert ex.value.win + ex.value.draw + ex.value.loss == pytest.approx(1.0)

    def test_policy_target_has_played_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        record = GameRecord(
            actions=(Action(move.from_square, move.to_square, move.promotion),),
            result=0.0,
        )
        encoder = SimpleBoardEncoder()
        examples = list(generate_examples_from_game(record, encoder))
        assert len(examples) == 1
        assert examples[0].policy.data[move.from_square, move.to_square].item() == 1.0
