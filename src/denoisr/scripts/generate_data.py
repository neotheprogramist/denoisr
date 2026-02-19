"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    boards:   Tensor[N, 12, 8, 8]
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)

Stockfish evaluation is parallelized across multiple worker processes,
each owning its own Stockfish subprocess.
"""

import atexit
import multiprocessing
import os
import argparse
import shutil
import sys
from pathlib import Path

import chess
import torch
from tqdm import tqdm

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

# -- Per-worker process globals (set by _init_worker) ---------------------

_oracle: StockfishOracle | None = None
_encoder: SimpleBoardEncoder | None = None


def _init_worker(stockfish_path: str, stockfish_depth: int) -> None:
    global _oracle, _encoder
    _oracle = StockfishOracle(path=stockfish_path, depth=stockfish_depth)
    _encoder = SimpleBoardEncoder()
    atexit.register(_oracle.close)


def _evaluate_position(
    fen: str,
) -> tuple[torch.Tensor, torch.Tensor, tuple[float, float, float]]:
    if _oracle is None or _encoder is None:
        raise RuntimeError("Worker not initialized")
    board = chess.Board(fen)
    board_tensor = _encoder.encode(board)
    policy, value, _ = _oracle.evaluate(board)
    return board_tensor.data, policy.data, (value.win, value.draw, value.loss)


# -- Public API -----------------------------------------------------------


def _extract_fens(pgn_path: Path, max_positions: int) -> list[str]:
    streamer = SimplePGNStreamer()
    fens: list[str] = []
    pbar = tqdm(total=max_positions, desc="Extracting positions", unit="pos")

    for record in streamer.stream(pgn_path):
        board = chess.Board()
        for action in record.actions:
            if len(fens) >= max_positions:
                break
            fens.append(board.fen())
            pbar.update(1)
            move = chess.Move(
                action.from_square,
                action.to_square,
                action.promotion,
            )
            board.push(move)

        if len(fens) >= max_positions:
            break

    pbar.close()
    return fens


def generate_examples(
    pgn_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
) -> list[TrainingExample]:
    fens = _extract_fens(pgn_path, max_examples)
    print(f"Extracted {len(fens)} positions, evaluating with {num_workers} workers")

    examples: list[TrainingExample] = []

    with multiprocessing.Pool(
        num_workers,
        initializer=_init_worker,
        initargs=(stockfish_path, stockfish_depth),
    ) as pool:
        results = pool.imap_unordered(_evaluate_position, fens)
        for board_data, policy_data, (win, draw, loss) in tqdm(
            results, total=len(fens), desc="Evaluating positions", unit="pos"
        ):
            examples.append(
                TrainingExample(
                    board=BoardTensor(board_data),
                    policy=PolicyTarget(policy_data),
                    value=ValueTarget(win=win, draw=draw, loss=loss),
                )
            )

    print(f"Generated {len(examples)} training examples")
    return examples


def stack_examples(
    examples: list[TrainingExample],
) -> dict[str, torch.Tensor]:
    boards = torch.stack([ex.board.data for ex in examples])
    policies = torch.stack([ex.policy.data for ex in examples])
    values = torch.tensor(
        [[ex.value.win, ex.value.draw, ex.value.loss] for ex in examples],
        dtype=torch.float32,
    )
    return {"boards": boards, "policies": policies, "values": values}


def unstack_examples(
    data: dict[str, torch.Tensor],
) -> list[TrainingExample]:
    boards = data["boards"]
    policies = data["policies"]
    values = data["values"]
    n = boards.shape[0]
    return [
        TrainingExample(
            board=BoardTensor(boards[i]),
            policy=PolicyTarget(policies[i]),
            value=ValueTarget(
                win=values[i, 0].item(),
                draw=values[i, 1].item(),
                loss=values[i, 2].item(),
            ),
        )
        for i in range(n)
    ]


def _default_num_workers() -> int:
    return (os.cpu_count() or 1) * 2 + 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate training data from PGN games with Stockfish evaluation"
    )
    parser.add_argument("--pgn", required=True, help="Path to .pgn or .pgn.zst file")
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (auto-detected from PATH if omitted)",
    )
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument(
        "--workers",
        type=int,
        default=_default_num_workers(),
        help=f"Worker processes (default: cpu_count*2+1 = {_default_num_workers()})",
    )
    parser.add_argument(
        "--output", type=str, default="outputs/training_data.pt"
    )
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        print(
            "Error: Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
            file=sys.stderr,
        )
        sys.exit(1)

    examples = generate_examples(
        Path(args.pgn),
        stockfish_path,
        args.stockfish_depth,
        args.max_examples,
        args.workers,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stack_examples(examples), output_path)
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
