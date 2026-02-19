"""Generate training data: PGN + Stockfish -> stacked tensors on disk.

Produces a .pt file with:
    boards:   Tensor[N, 12, 8, 8]
    policies: Tensor[N, 64, 64]
    values:   Tensor[N, 3]        (win, draw, loss per example)
"""

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


def generate_examples(
    pgn_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
) -> list[TrainingExample]:
    streamer = SimplePGNStreamer()
    encoder = SimpleBoardEncoder()
    examples: list[TrainingExample] = []

    pbar = tqdm(total=max_examples, desc="Generating examples", unit="pos")

    with StockfishOracle(path=stockfish_path, depth=stockfish_depth) as oracle:
        for record in streamer.stream(pgn_path):
            board = chess.Board()
            for action in record.actions:
                if len(examples) >= max_examples:
                    break
                board_tensor = encoder.encode(board)
                policy_target, value_target, _ = oracle.evaluate(board)
                examples.append(
                    TrainingExample(
                        board=board_tensor,
                        policy=policy_target,
                        value=value_target,
                    )
                )
                pbar.update(1)
                move = chess.Move(
                    action.from_square,
                    action.to_square,
                    action.promotion,
                )
                board.push(move)

            if len(examples) >= max_examples:
                break

    pbar.close()
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
        Path(args.pgn), stockfish_path, args.stockfish_depth, args.max_examples
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(stack_examples(examples), output_path)
    print(f"Saved {len(examples)} examples to {output_path}")


if __name__ == "__main__":
    main()
