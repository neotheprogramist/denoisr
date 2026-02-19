"""Board color-flip augmentation for training data.

Flips a chess position by mirroring ranks and swapping colors.
This doubles the effective dataset: every White-to-play position
becomes an equivalent Black-to-play position (and vice versa).
"""

import torch
from torch import Tensor

# Precomputed square flip: rank r -> rank 7-r, file unchanged
# Square index = rank*8 + file. Flipped = (7-rank)*8 + file.
_SQUARE_FLIP = torch.tensor([(7 - (i // 8)) * 8 + (i % 8) for i in range(64)])


def flip_board(board: Tensor, num_planes: int) -> Tensor:
    """Flip board tensor [C, 8, 8]: mirror ranks, swap white/black planes.

    Handles both 12-plane (simple) and 110-plane (extended) encoders.
    """
    flipped = board.clone()
    # Mirror ranks (rank 0 <-> rank 7)
    flipped = flipped.flip(1)

    # Swap white/black piece planes in groups of 12
    # Planes 0-5 = white pieces, 6-11 = black pieces
    num_piece_groups = min(num_planes, 96) // 12
    for g in range(num_piece_groups):
        offset = g * 12
        white = flipped[offset : offset + 6].clone()
        black = flipped[offset + 6 : offset + 12].clone()
        flipped[offset : offset + 6] = black
        flipped[offset + 6 : offset + 12] = white

    # Extended encoder metadata planes (starting at plane 96)
    if num_planes > 96:
        meta = 96
        # Castling: swap white (meta+0, meta+1) <-> black (meta+2, meta+3)
        wk, wq = flipped[meta].clone(), flipped[meta + 1].clone()
        flipped[meta] = flipped[meta + 2]
        flipped[meta + 1] = flipped[meta + 3]
        flipped[meta + 2] = wk
        flipped[meta + 3] = wq
        # En passant (meta+4): already rank-flipped above
        # Rule50 (meta+5): unchanged
        # Repetition (meta+6): unchanged
        # Side to move (meta+7): invert
        flipped[meta + 7] = 1.0 - flipped[meta + 7]
        # Material: swap white (meta+8) <-> black (meta+9)
        w_mat = flipped[meta + 8].clone()
        flipped[meta + 8] = flipped[meta + 9]
        flipped[meta + 9] = w_mat
        # Check attackers: swap white (meta+10) <-> black (meta+11)
        w_chk = flipped[meta + 10].clone()
        flipped[meta + 10] = flipped[meta + 11]
        flipped[meta + 11] = w_chk
        # Bishops (meta+12, meta+13): unchanged (symmetric property)

    return flipped


def flip_policy(policy: Tensor) -> Tensor:
    """Flip policy tensor [64, 64]: mirror source and destination squares."""
    return policy[_SQUARE_FLIP][:, _SQUARE_FLIP]


def flip_value(win: float, draw: float, loss: float) -> tuple[float, float, float]:
    """Swap win and loss probabilities."""
    return loss, draw, win
