"""Phase 1: Supervised training from pre-generated training data.

Pipeline:
    training_data.pt -> TrainingExamples -> SupervisedTrainer

Gate to Phase 2: policy top-1 accuracy > 30% on held-out positions.
"""

import argparse
import random
import time
from pathlib import Path

import torch
from torch.amp import autocast  # type: ignore[attr-defined]
from torch.utils.data import DataLoader
from tqdm import tqdm

from denoisr.scripts.config import (
    add_model_args,
    add_training_args,
    build_backbone,
    build_encoder,
    build_policy_head,
    build_value_head,
    detect_device,
    load_checkpoint,
    maybe_compile,
    resolve_gradient_checkpointing,
    save_checkpoint,
    training_config_from_args,
)
from denoisr.scripts.generate_data import unstack_examples
from denoisr.training.dataset import ChessDataset
from denoisr.training.logger import TrainingLogger
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import TrainingExample


def measure_accuracy(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
) -> tuple[float, float]:
    correct_1 = 0
    correct_5 = 0
    total = 0

    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    autocast_enabled = device.type == "cuda"

    with torch.no_grad(), autocast(autocast_device, enabled=autocast_enabled):
        for ex in examples:
            board = ex.board.data.unsqueeze(0).to(device)
            latent = trainer.encoder(board)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features).squeeze(0)

            pred_flat = logits.reshape(-1)
            target_flat = ex.policy.data.reshape(-1)
            target_idx = target_flat.argmax().item()

            top5 = pred_flat.topk(5).indices.tolist()
            if top5[0] == target_idx:
                correct_1 += 1
            if target_idx in top5:
                correct_5 += 1
            total += 1

    return correct_1 / max(total, 1), correct_5 / max(total, 1)


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
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="TensorBoard run name (default: timestamp)",
    )
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    device = detect_device()
    tcfg = training_config_from_args(args)
    print(f"Device: {device}")

    # --- Load checkpoint ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    print(f"Loaded checkpoint: d_s={cfg.d_s}, heads={cfg.num_heads}, layers={cfg.num_layers}")

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])

    encoder = maybe_compile(encoder, device)
    backbone = maybe_compile(backbone, device)
    policy_head = maybe_compile(policy_head, device)
    value_head = maybe_compile(value_head, device)

    # --- Load pre-generated data ---
    raw = torch.load(Path(args.data), weights_only=True)
    all_examples = unstack_examples(raw)
    print(f"Loaded {len(all_examples)} training examples from {args.data}")
    random.shuffle(all_examples)

    holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
    holdout = all_examples[:holdout_n]
    train = all_examples[holdout_n:]
    print(f"Train: {len(train)}, Holdout: {holdout_n}")

    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        use_harmony_dream=tcfg.use_harmony_dream,
        harmony_ema_decay=tcfg.harmony_ema_decay,
    )

    trainer = SupervisedTrainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
        total_epochs=args.epochs,
        warmup_epochs=tcfg.warmup_epochs,
        max_grad_norm=tcfg.max_grad_norm,
        weight_decay=tcfg.weight_decay,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        min_lr=tcfg.min_lr,
    )

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        # --- Build DataLoader from stacked tensors ---
        bs = args.batch_size
        train_boards = torch.stack([ex.board.data for ex in train])
        train_policies = torch.stack([ex.policy.data for ex in train])
        train_values = torch.tensor(
            [[ex.value.win, ex.value.draw, ex.value.loss] for ex in train],
            dtype=torch.float32,
        )

        train_dataset = ChessDataset(
            train_boards,
            train_policies,
            train_values,
            num_planes=cfg.num_planes,
            augment=True,
        )
        train_loader: DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = (
            DataLoader(
                train_dataset,
                batch_size=bs,
                shuffle=True,
                num_workers=tcfg.num_workers,
                pin_memory=(device.type == "cuda"),
                persistent_workers=True,
            )
        )

        # --- Train ---
        best_acc = 0.0

        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": bs,
                "epochs": args.epochs,
                "d_s": cfg.d_s,
                "num_heads": cfg.num_heads,
                "num_layers": cfg.num_layers,
                "ffn_dim": cfg.ffn_dim,
                "num_planes": cfg.num_planes,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "max_grad_norm": tcfg.max_grad_norm,
                "weight_decay": tcfg.weight_decay,
                "encoder_lr_multiplier": tcfg.encoder_lr_multiplier,
                "min_lr": tcfg.min_lr,
                "warmup_epochs": tcfg.warmup_epochs,
                "policy_weight": tcfg.policy_weight,
                "value_weight": tcfg.value_weight,
                "use_harmony_dream": tcfg.use_harmony_dream,
                "num_workers": tcfg.num_workers,
            },
            {"best_top1": 0.0},
        )

        global_step = 0

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
            )
            for boards_batch, policies_batch, values_batch in pbar:
                loss, breakdown = trainer.train_step_tensors(
                    boards_batch, policies_batch, values_batch
                )
                logger.log_train_step(global_step, loss, breakdown)
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    policy=f"{breakdown['policy']:.4f}",
                    value=f"{breakdown['value']:.4f}",
                )
            pbar.close()
            trainer.scheduler_step()

            epoch_duration = time.monotonic() - epoch_start
            samples_per_sec = len(train) / epoch_duration
            avg_loss = epoch_loss / max(num_batches, 1)
            top1, top5 = measure_accuracy(trainer, holdout, device)
            current_lr = trainer.optimizer.param_groups[0]["lr"]

            logger.log_epoch(epoch, avg_loss, top1, top5, current_lr)
            logger.log_epoch_timing(epoch, epoch_duration, samples_per_sec)

            print(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"avg_loss={avg_loss:.4f} top1={top1:.1%} top5={top5:.1%}"
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
            if top1 > tcfg.phase1_gate:
                print(f"PHASE 1 GATE PASSED: top-1 accuracy {top1:.1%} > {tcfg.phase1_gate:.0%}")
                print("Ready for Phase 2.")
                break

    print(f"Best top-1 accuracy: {best_acc:.1%}")


if __name__ == "__main__":
    main()
