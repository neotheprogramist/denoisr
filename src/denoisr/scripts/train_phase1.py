"""Phase 1: Supervised training on Lichess games with Stockfish targets.

Pipeline:
    PGN file -> PGNStreamer -> StockfishOracle -> TrainingExamples -> SupervisedTrainer

Gate to Phase 2: policy top-1 accuracy > 30% on held-out positions.
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import chess
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.dataset import generate_examples_from_game
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.data.stockfish_oracle import StockfishOracle
from denoisr.scripts.config import (
    add_model_args,
    build_backbone,
    build_encoder,
    build_policy_head,
    build_value_head,
    config_from_args,
    detect_device,
    save_checkpoint,
)
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import TrainingExample


def generate_data(
    pgn_path: str,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
) -> list[TrainingExample]:
    streamer = SimplePGNStreamer()
    encoder = SimpleBoardEncoder()
    examples: list[TrainingExample] = []

    print(f"Generating examples from {pgn_path} with Stockfish depth={stockfish_depth}...")

    with StockfishOracle(path=stockfish_path, depth=stockfish_depth) as oracle:
        for i, record in enumerate(streamer.stream(pgn_path)):
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
                move = chess.Move(
                    action.from_square,
                    action.to_square,
                    action.promotion,
                )
                board.push(move)

            if len(examples) >= max_examples:
                break
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} games, {len(examples)} examples")

    print(f"Generated {len(examples)} training examples")
    return examples


def measure_top1(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
) -> float:
    correct = 0
    total = 0
    for m in (trainer.encoder, trainer.backbone, trainer.policy_head):
        m.training = False

    with torch.no_grad():
        for ex in examples:
            board = ex.board.data.unsqueeze(0).to(device)
            latent = trainer.encoder(board)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features).squeeze(0)

            pred = logits.reshape(-1).argmax().item()
            target = ex.policy.data.reshape(-1).argmax().item()
            if pred == target:
                correct += 1
            total += 1

    return correct / max(total, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Supervised training")
    parser.add_argument("--pgn", required=True, help="Path to .pgn or .pgn.zst file")
    parser.add_argument(
        "--stockfish",
        default=None,
        help="Path to Stockfish binary (auto-detected from PATH if omitted)",
    )
    parser.add_argument("--stockfish-depth", type=int, default=10)
    parser.add_argument("--max-examples", type=int, default=100_000)
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/phase1.pt")
    parser.add_argument("--log-every", type=int, default=10, help="Log every N batches")
    add_model_args(parser)
    args = parser.parse_args()

    stockfish_path = args.stockfish or shutil.which("stockfish")
    if stockfish_path is None:
        print(
            "Error: Stockfish not found. Install it or pass --stockfish /path/to/stockfish",
            file=sys.stderr,
        )
        sys.exit(1)
    args.stockfish = stockfish_path

    cfg = config_from_args(args)
    device = detect_device()
    print(f"Device: {device}")
    print(f"Model config: d_s={cfg.d_s}, heads={cfg.num_heads}, layers={cfg.num_layers}")

    # --- Generate data ---
    all_examples = generate_data(
        args.pgn, args.stockfish, args.stockfish_depth, args.max_examples
    )
    random.shuffle(all_examples)

    holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
    holdout = all_examples[:holdout_n]
    train = all_examples[holdout_n:]
    print(f"Train: {len(train)}, Holdout: {holdout_n}")

    # --- Build model ---
    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)
    loss_fn = ChessLossComputer(use_harmony_dream=True)

    trainer = SupervisedTrainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
    )

    # --- Train ---
    bs = args.batch_size
    best_acc = 0.0

    for epoch in range(args.epochs):
        random.shuffle(train)
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(train), bs):
            batch = train[i : i + bs]
            if not batch:
                break
            loss, breakdown = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1

            if args.log_every and num_batches % args.log_every == 0:
                print(
                    f"  [{epoch+1}/{args.epochs}] batch {num_batches}: "
                    f"loss={loss:.4f} policy={breakdown['policy']:.4f} "
                    f"value={breakdown['value']:.4f}"
                )

        avg_loss = epoch_loss / max(num_batches, 1)

        top1 = measure_top1(trainer, holdout, device)

        print(
            f"Epoch {epoch+1}/{args.epochs}: "
            f"avg_loss={avg_loss:.4f} top1_accuracy={top1:.1%}"
        )

        if top1 > best_acc:
            best_acc = top1
            save_checkpoint(
                Path(args.output),
                cfg,
                encoder=encoder.state_dict(),
                backbone=backbone.state_dict(),
                policy_head=policy_head.state_dict(),
                value_head=value_head.state_dict(),
                optimizer=trainer.optimizer.state_dict(),
            )

        # Phase gate check
        if top1 > 0.30:
            print(f"PHASE 1 GATE PASSED: top-1 accuracy {top1:.1%} > 30%")
            print("Ready for Phase 2.")
            break

    print(f"Best top-1 accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
