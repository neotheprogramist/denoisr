"""UCI (Universal Chess Interface) protocol parser and formatter.

Handles parsing UCI commands from stdin and formatting responses.
The actual engine logic is in engine.py.
"""


def parse_position(command: str) -> tuple[str | None, list[str]]:
    """Parse a UCI 'position' command.

    Returns (fen_or_none, list_of_uci_moves).
    """
    parts = command.split()
    fen = None
    moves: list[str] = []

    if "startpos" in parts:
        if "moves" in parts:
            moves_idx = parts.index("moves")
            moves = parts[moves_idx + 1 :]
    elif "fen" in parts:
        fen_idx = parts.index("fen")
        if "moves" in parts:
            moves_idx = parts.index("moves")
            fen = " ".join(parts[fen_idx + 1 : moves_idx])
            moves = parts[moves_idx + 1 :]
        else:
            fen = " ".join(parts[fen_idx + 1 :])

    return fen, moves


def parse_go(command: str) -> dict[str, int | bool]:
    """Parse a UCI 'go' command into parameters."""
    parts = command.split()
    params: dict[str, int | bool] = {}

    int_keys = {
        "movetime",
        "depth",
        "nodes",
        "wtime",
        "btime",
        "winc",
        "binc",
        "movestogo",
    }

    i = 1
    while i < len(parts):
        token = parts[i]
        if token == "infinite":
            params["infinite"] = True
        elif token in int_keys and i + 1 < len(parts):
            params[token] = int(parts[i + 1])
            i += 1
        i += 1

    return params


def format_bestmove(uci_move: str) -> str:
    """Format a UCI bestmove response."""
    return f"bestmove {uci_move}"


def run_uci_loop(
    engine_select_move_fn: object,
) -> None:
    """Main UCI loop reading from stdin.

    Connect to any UCI-compatible GUI (CuteChess, Arena, etc.).
    engine_select_move_fn: callable(chess.Board) -> chess.Move.
    """
    import sys

    import chess

    board = chess.Board()

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        if line == "uci":
            print("id name denoisr")
            print("id author denoisr-team")
            print("uciok")

        elif line == "isready":
            print("readyok")

        elif line.startswith("position"):
            fen, moves = parse_position(line)
            board = chess.Board(fen) if fen else chess.Board()
            for uci in moves:
                board.push_uci(uci)

        elif line.startswith("go"):
            move = engine_select_move_fn(board)
            print(format_bestmove(move.uci()))

        elif line == "quit":
            break

        sys.stdout.flush()
