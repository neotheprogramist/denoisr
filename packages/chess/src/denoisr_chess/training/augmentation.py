"""Training-time augmentations for chess data.

Includes board color-flip (mirror ranks + swap colors), value noise
injection, and policy temperature scaling.
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

    # Extended tactical planes (starting at plane 110)
    if num_planes > 110:
        tac = 110
        # 6 white/black pairs that must be swapped on color flip:
        # attacks(0,1), defense(2,3), hanging(4,5), pinned(6,7),
        # mobility(8,9), threats(10,11)
        for offset in range(0, 12, 2):
            w_plane = flipped[tac + offset].clone()
            flipped[tac + offset] = flipped[tac + offset + 1]
            flipped[tac + offset + 1] = w_plane

    return flipped


def flip_policy(policy: Tensor) -> Tensor:
    """Flip policy tensor [64, 64]: mirror source and destination squares."""
    idx = _SQUARE_FLIP.to(policy.device)
    return policy[idx][:, idx]


def flip_value(win: float, draw: float, loss: float) -> tuple[float, float, float]:
    """Swap win and loss probabilities."""
    return loss, draw, win


def augment_value_noise(value: Tensor, scale: float = 0.02) -> Tensor:
    """Add Gaussian noise to WDL targets and re-normalize to valid distribution."""
    if scale == 0.0:
        return value
    noise = torch.randn_like(value) * scale
    noisy = (value + noise).clamp(min=0.0)
    total = noisy.sum()
    return noisy / total if total > 0 else value


def augment_policy_temperature(
    policy: Tensor, temp_min: float = 0.8, temp_max: float = 1.2
) -> Tensor:
    """Apply random temperature scaling to policy distribution.

    Only affects nonzero entries (legal moves). Temperature < 1 sharpens,
    temperature > 1 flattens the distribution.
    """
    temp = temp_min + torch.rand(1).item() * (temp_max - temp_min)
    mask = policy > 0
    if not mask.any():
        return policy
    result = policy.clone()
    result[mask] = result[mask] ** (1.0 / temp)
    total = result.sum()
    return result / total if total > 0 else policy
