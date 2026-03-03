import chess
import chess.engine
import torch

from denoisr.types import PolicyTarget, ValueTarget


class StockfishOracle:
    def __init__(
        self,
        path: str,
        depth: int = 12,
        policy_temperature: float = 80.0,
        label_smoothing: float = 0.02,
    ) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._engine.configure({"UCI_ShowWDL": True})
        self._depth = depth
        self._policy_temperature = policy_temperature
        self._label_smoothing = label_smoothing

    def evaluate(self, board: chess.Board) -> tuple[PolicyTarget, ValueTarget, float]:
        policy = self._get_policy(board)
        value, cp = self._get_value(board)
        return policy, value, cp

    def _get_policy(self, board: chess.Board) -> PolicyTarget:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PolicyTarget(torch.zeros(64, 64, dtype=torch.float32))

        scores: list[float] = []
        for move in legal_moves:
            board.push(move)
            info = self._engine.analyse(
                board, chess.engine.Limit(depth=max(1, self._depth - 2))
            )
            score = info["score"].white()
            cp_val = score.score(mate_score=10000)
            if cp_val is None:
                cp_val = 0
            # After board.push(move) the turn has flipped, so analyse()
            # returns score from the current side's perspective. Negate to
            # convert back to "from the side that just moved" perspective.
            if board.turn == chess.WHITE:
                cp_val = -cp_val
            scores.append(float(cp_val))
            board.pop()

        # Softmax over centipawn scores to get distribution
        t = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(t / self._policy_temperature, dim=0)
        if self._label_smoothing > 0:
            n_legal = len(legal_moves)
            probs = (
                1 - self._label_smoothing
            ) * probs + self._label_smoothing / n_legal

        data = torch.zeros(64, 64, dtype=torch.float32)
        for move, prob in zip(legal_moves, probs):
            data[move.from_square, move.to_square] = prob.item()

        return PolicyTarget(data)

    def _get_value(self, board: chess.Board) -> tuple[ValueTarget, float]:
        # Terminal positions have deterministic WDL; Stockfish omits WDL for
        # these, so resolve them from the board state directly.
        if board.is_checkmate():
            # The side to move has been checkmated.
            if board.turn == chess.WHITE:
                return ValueTarget(win=0.0, draw=0.0, loss=1.0), -10000.0
            return ValueTarget(win=1.0, draw=0.0, loss=0.0), 10000.0
        if board.is_stalemate() or board.is_insufficient_material() or board.can_claim_draw():
            return ValueTarget(win=0.0, draw=1.0, loss=0.0), 0.0

        info = self._engine.analyse(board, chess.engine.Limit(depth=self._depth))
        score = info["score"].white()
        cp_val = score.score(mate_score=10000)
        if cp_val is None:
            cp_val = 0

        wdl = info.get("wdl")
        if wdl is None:
            raise ValueError(
                "Stockfish did not return WDL data. "
                "Requires Stockfish 14+ compiled with WDL support. "
                "Check your Stockfish binary version."
            )
        wdl_white = wdl.white()
        total = wdl_white.wins + wdl_white.draws + wdl_white.losses
        value = ValueTarget(
            win=wdl_white.wins / total,
            draw=wdl_white.draws / total,
            loss=wdl_white.losses / total,
        )

        return value, float(cp_val)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishOracle":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
