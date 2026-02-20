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
SMALL_NUM_HEADS = 4
SMALL_NUM_LAYERS = 2
SMALL_FFN_DIM = 128
SMALL_NUM_TIMESTEPS = 20


@pytest.fixture
def small_latent(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 64, SMALL_D_S, device=device)


@pytest.fixture
def small_board_tensor(device: torch.device) -> torch.Tensor:
    return torch.randn(2, 12, 8, 8, device=device)


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
