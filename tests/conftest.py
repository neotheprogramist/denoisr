import chess
import pytest
import torch
from hypothesis import strategies as st


@pytest.fixture
def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


SMALL_D_S = 64


@st.composite
def random_boards(draw: st.DrawFn) -> chess.Board:
    board = chess.Board()
    num_moves = draw(st.integers(min_value=0, max_value=60))
    for _ in range(num_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = draw(st.sampled_from(legal))
        board.push(move)
    return board
