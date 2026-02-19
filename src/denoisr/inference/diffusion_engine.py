import chess
import torch
from torch import nn

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.nn.diffusion import CosineNoiseSchedule


class DiffusionChessEngine:
    """DiffuSearch-style chess engine with diffusion-enhanced inference.

    Combines encoder + diffusion imagination + policy backbone:
    1. Encode current board to latent state
    2. Run N denoising steps to imagine future trajectory
    3. Fuse current latent with denoised future
    4. Run policy backbone + head on fused representation

    The num_denoising_steps parameter gives anytime search:
    more steps = stronger but slower inference.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        diffusion: nn.Module,
        schedule: CosineNoiseSchedule,
        board_encoder: SimpleBoardEncoder,
        device: torch.device | None = None,
        num_denoising_steps: int = 10,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._diffusion = diffusion
        self._schedule = schedule
        self._board_encoder = board_encoder
        self._device = device or torch.device("cpu")
        self._num_steps = num_denoising_steps

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self._set_eval()
        latent = self._encode_board(board)
        enriched = self._diffusion_imagine(latent)
        features = self._backbone(enriched)
        logits = self._policy_head(features).squeeze(0)

        legal_mask = torch.full((64, 64), float("-inf"))
        for move in board.legal_moves:
            legal_mask[move.from_square, move.to_square] = 0.0
        legal_mask = legal_mask.to(self._device)

        probs = torch.softmax((logits + legal_mask).reshape(-1), dim=0)
        idx = torch.multinomial(probs, 1).item()
        from_sq, to_sq = idx // 64, idx % 64

        promotion = None
        piece = board.piece_at(from_sq)
        if piece and piece.piece_type == chess.PAWN:
            to_rank = chess.square_rank(to_sq)
            if (piece.color == chess.WHITE and to_rank == 7) or (
                piece.color == chess.BLACK and to_rank == 0
            ):
                promotion = chess.QUEEN

        return chess.Move(from_sq, to_sq, promotion)

    @torch.no_grad()
    def evaluate(self, board: chess.Board) -> tuple[float, float, float]:
        self._set_eval()
        latent = self._encode_board(board)
        enriched = self._diffusion_imagine(latent)
        features = self._backbone(enriched)
        wdl, _ = self._value_head(features)
        wdl = wdl.squeeze(0)
        return (wdl[0].item(), wdl[1].item(), wdl[2].item())

    def _encode_board(self, board: chess.Board) -> torch.Tensor:
        board_tensor = self._board_encoder.encode(board).data
        return self._encoder(board_tensor.unsqueeze(0).to(self._device))

    def _diffusion_imagine(self, latent: torch.Tensor) -> torch.Tensor:
        """Run iterative denoising to imagine future trajectories."""
        x = torch.randn_like(latent)
        step_size = max(1, self._schedule.num_timesteps // self._num_steps)

        for i in range(self._num_steps):
            t_val = max(0, self._schedule.num_timesteps - 1 - i * step_size)
            t = torch.tensor([t_val], device=self._device)
            noise_pred = self._diffusion(x, t, latent)
            ab = self._schedule.alpha_bar.to(self._device)[t_val]
            x = (x - (1 - ab).sqrt() * noise_pred) / ab.sqrt()

        return (latent + x) / 2

    def _set_eval(self) -> None:
        self._encoder.eval()
        self._backbone.eval()
        self._policy_head.eval()
        self._value_head.eval()
        self._diffusion.eval()
