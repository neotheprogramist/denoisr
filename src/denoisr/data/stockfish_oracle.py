import chess
import chess.engine
import torch

from denoisr.types import PolicyTarget, ValueTarget


class StockfishOracle:
    def __init__(self, path: str, depth: int = 12) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._depth = depth

    def evaluate(
        self, board: chess.Board
    ) -> tuple[PolicyTarget, ValueTarget, float]:
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
            if board.turn == chess.WHITE:
                cp_val = -cp_val
            scores.append(float(cp_val))
            board.pop()

        # Softmax over centipawn scores to get distribution
        t = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(t / 100.0, dim=0)

        data = torch.zeros(64, 64, dtype=torch.float32)
        for move, prob in zip(legal_moves, probs):
            data[move.from_square, move.to_square] = prob.item()

        return PolicyTarget(data)

    def _get_value(self, board: chess.Board) -> tuple[ValueTarget, float]:
        info = self._engine.analyse(
            board, chess.engine.Limit(depth=self._depth)
        )
        score = info["score"].white()
        cp_val = score.score(mate_score=10000)
        if cp_val is None:
            cp_val = 0

        wdl = info.get("wdl")
        if wdl is not None:
            wdl_white = wdl.white()
            total = wdl_white.wins + wdl_white.draws + wdl_white.losses
            value = ValueTarget(
                win=wdl_white.wins / total,
                draw=wdl_white.draws / total,
                loss=wdl_white.losses / total,
            )
        else:
            # Approximate WDL from centipawns using sigmoid
            win_prob = 1.0 / (1.0 + 10.0 ** (-float(cp_val) / 400.0))
            value = ValueTarget(
                win=win_prob, draw=0.0, loss=1.0 - win_prob
            )

        return value, float(cp_val)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishOracle":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
