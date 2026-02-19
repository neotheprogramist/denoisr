"""Phase 1: Supervised training from pre-generated training data.

Pipeline:
    training_data.pt -> TrainingExamples -> SupervisedTrainer

Gate to Phase 2: policy top-1 accuracy > 30% on held-out positions.
"""

import argparse
import random
from pathlib import Path

import torch
from tqdm import tqdm

from denoisr.scripts.config import (
    add_model_args,
    build_backbone,
    build_encoder,
    build_policy_head,
    build_value_head,
    detect_device,
    load_checkpoint,
    save_checkpoint,
)
from denoisr.scripts.generate_data import unstack_examples
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import TrainingExample


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
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint to load (create with denoisr-init)",
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to training data .pt file (create with denoisr-generate-data)",
    )
    parser.add_argument("--holdout-frac", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/phase1.pt")
    add_model_args(parser)
    args = parser.parse_args()

    device = detect_device()
    print(f"Device: {device}")

    # --- Load checkpoint ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    print(f"Loaded checkpoint: d_s={cfg.d_s}, heads={cfg.num_heads}, layers={cfg.num_layers}")

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])

    # --- Load pre-generated data ---
    raw = torch.load(Path(args.data), weights_only=True)
    all_examples = unstack_examples(raw)
    print(f"Loaded {len(all_examples)} training examples from {args.data}")
    random.shuffle(all_examples)

    holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
    holdout = all_examples[:holdout_n]
    train = all_examples[holdout_n:]
    print(f"Train: {len(train)}, Holdout: {holdout_n}")

    loss_fn = ChessLossComputer(use_harmony_dream=True)

    trainer = SupervisedTrainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
        total_epochs=args.epochs,
        warmup_epochs=3,
    )

    # --- Train ---
    bs = args.batch_size
    best_acc = 0.0

    for epoch in range(args.epochs):
        random.shuffle(train)
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            range(0, len(train), bs),
            desc=f"Epoch {epoch+1}/{args.epochs}",
            leave=False,
            smoothing=0.3,
        )
        for i in pbar:
            batch = train[i : i + bs]
            if not batch:
                break
            loss, breakdown = trainer.train_step(batch)
            epoch_loss += loss
            num_batches += 1
            pbar.set_postfix(
                loss=f"{loss:.4f}",
                policy=f"{breakdown['policy']:.4f}",
                value=f"{breakdown['value']:.4f}",
            )
        pbar.close()
        trainer.scheduler_step()

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
