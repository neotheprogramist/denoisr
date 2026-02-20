"""Minimal UCI engine for testing. Responds to uci/isready/go with fixed moves."""

import sys

import chess


def main() -> None:
    board = chess.Board()
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        if line == "uci":
            print("id name MockEngine")
            print("id author test")
            print("uciok")
        elif line == "isready":
            print("readyok")
        elif line.startswith("position"):
            parts = line.split()
            if "startpos" in parts:
                board = chess.Board()
            elif "fen" in parts:
                fen_idx = parts.index("fen")
                if "moves" in parts:
                    moves_idx = parts.index("moves")
                    fen = " ".join(parts[fen_idx + 1 : moves_idx])
                else:
                    moves_idx = len(parts)
                    fen = " ".join(parts[fen_idx + 1 :])
                board = chess.Board(fen)
            if "moves" in parts:
                moves_idx = parts.index("moves")
                for uci in parts[moves_idx + 1 :]:
                    board.push_uci(uci)
        elif line.startswith("go"):
            move = next(iter(board.legal_moves))
            print(f"bestmove {move.uci()}")
        elif line == "quit":
            break
        sys.stdout.flush()


if __name__ == "__main__":
    main()
