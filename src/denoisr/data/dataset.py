from collections.abc import Iterator

import chess
import torch
from torch.utils.data import Dataset

from denoisr.data.protocols import BoardEncoder
from denoisr.types import (
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


class ChessDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[TrainingExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self._examples[idx]
        value = torch.tensor(
            [ex.value.win, ex.value.draw, ex.value.loss],
            dtype=torch.float32,
        )
        return ex.board.data, ex.policy.data, value


def generate_examples_from_game(
    record: GameRecord, encoder: BoardEncoder
) -> Iterator[TrainingExample]:
    board = chess.Board()
    for action in record.actions:
        board_tensor = encoder.encode(board)

        policy_data = torch.zeros(64, 64, dtype=torch.float32)
        policy_data[action.from_square, action.to_square] = 1.0
        policy = PolicyTarget(policy_data)

        if record.result == 1.0:
            value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        elif record.result == -1.0:
            value = ValueTarget(win=0.0, draw=0.0, loss=1.0)
        else:
            value = ValueTarget(win=0.0, draw=1.0, loss=0.0)

        yield TrainingExample(board=board_tensor, policy=policy, value=value)

        move = chess.Move(
            action.from_square, action.to_square, action.promotion
        )
        board.push(move)
