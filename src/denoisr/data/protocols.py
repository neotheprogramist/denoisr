from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import chess

from denoisr.types import (
    Action,
    BoardTensor,
    GameRecord,
    PolicyTarget,
    ValueTarget,
)


class BoardEncoder(Protocol):
    @property
    def num_planes(self) -> int: ...
    def encode(self, board: chess.Board) -> BoardTensor: ...


class ActionEncoder(Protocol):
    def encode_move(self, move: chess.Move) -> Action: ...
    def decode_action(self, action: Action, board: chess.Board) -> chess.Move: ...
    def action_to_index(self, action: Action) -> int: ...
    def index_to_action(self, index: int, board: chess.Board) -> Action: ...


class PGNStreamer(Protocol):
    def stream(self, path: Path) -> Iterator[GameRecord]: ...


class Oracle(Protocol):
    def evaluate(
        self, board: chess.Board
    ) -> tuple[PolicyTarget, ValueTarget, float]: ...
