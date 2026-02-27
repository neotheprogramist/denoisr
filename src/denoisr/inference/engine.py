import chess
import torch
from torch import nn

from denoisr.data.protocols import BoardEncoder


class ChessEngine:
    """Combines encoder + backbone + heads to select chess moves.

    Single-pass inference (no diffusion or MCTS). The simplest
    inference mode. Diffusion-enhanced and MCTS-enhanced modes
    can be added by extending this class.
    """

    def __init__(
        self,
        encoder: nn.Module,
        backbone: nn.Module,
        policy_head: nn.Module,
        value_head: nn.Module,
        board_encoder: BoardEncoder,
        device: torch.device | None = None,
    ) -> None:
        self._encoder = encoder
        self._backbone = backbone
        self._policy_head = policy_head
        self._value_head = value_head
        self._board_encoder = board_encoder
        self._device = device or torch.device("cpu")

    @torch.no_grad()
    def select_move(self, board: chess.Board) -> chess.Move:
        self._encoder.eval()
        self._backbone.eval()
        self._policy_head.eval()

        board_tensor = self._board_encoder.encode(board).data
        x = board_tensor.unsqueeze(0).to(self._device)

        latent = self._encoder(x)
        features = self._backbone(latent)
        logits = self._policy_head(features).squeeze(0)

        flat_logits = logits.reshape(-1)
        legal_flat = torch.zeros(64 * 64, dtype=torch.bool, device=self._device)
        for move in board.legal_moves:
            legal_flat[(move.from_square * 64) + move.to_square] = True
        if not bool(legal_flat.any().item()):
            raise ValueError("No legal moves available for current board position")

        masked_logits = flat_logits.masked_fill(~legal_flat, float("-inf"))
        probs = torch.softmax(masked_logits, dim=0)
        idx = int(torch.multinomial(probs, 1).item())

        from_sq = idx // 64
        to_sq = idx % 64

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
        self._encoder.eval()
        self._backbone.eval()
        self._value_head.eval()

        board_tensor = self._board_encoder.encode(board).data
        x = board_tensor.unsqueeze(0).to(self._device)

        latent = self._encoder(x)
        features = self._backbone(latent)
        wdl, _ = self._value_head.infer(features)
        wdl = wdl.squeeze(0)

        return (wdl[0].item(), wdl[1].item(), wdl[2].item())
