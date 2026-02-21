"""Phase 1: Supervised training from pre-generated training data.

Pipeline:
    training_data.pt -> TrainingExamples -> SupervisedTrainer

Gate to Phase 2: policy top-1 accuracy > 30% on held-out positions.
"""

import argparse
import logging
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
from denoisr.data.holdout_splitter import StratifiedHoldoutSplitter
from denoisr.scripts.generate_data import unstack_examples
from denoisr.training.dataset import ChessDataset
from denoisr.training.grok_tracker import GrokTracker
from denoisr.training.grokfast import GrokfastFilter
from denoisr.training.logger import TrainingLogger
from denoisr.training.loss import ChessLossComputer
from denoisr.training.resource_monitor import ResourceMonitor
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import TrainingExample

log = logging.getLogger(__name__)


def measure_accuracy(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    autocast_enabled = device.type == "cuda"

    correct_1 = 0
    correct_5 = 0
    total = len(examples)

    with torch.no_grad(), autocast(autocast_device, enabled=autocast_enabled):
        for i in range(0, total, batch_size):
            batch = examples[i : i + batch_size]
            boards = torch.stack([ex.board.data for ex in batch]).to(device)
            targets = torch.stack([ex.policy.data for ex in batch]).to(device)

            latent = trainer.encoder(boards)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features)

            pred_flat = logits.reshape(len(batch), -1)
            target_flat = targets.reshape(len(batch), -1)
            legal_mask = target_flat > 0
            masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
            target_idx = target_flat.argmax(dim=-1)  # (B,)

            top5 = masked_logits.topk(5, dim=-1).indices  # (B, 5)
            correct_1 += (top5[:, 0] == target_idx).sum().item()
            correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()

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
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
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

    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    device = detect_device()
    tcfg = training_config_from_args(args)
    use_tqdm = args.tqdm
    log.info("device=%s", device)

    # --- Load checkpoint ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    log.info("checkpoint loaded  d_s=%d  heads=%d  layers=%d", cfg.d_s, cfg.num_heads, cfg.num_layers)

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
    log.info("examples=%d  source=%s", len(all_examples), args.data)
    random.shuffle(all_examples)

    holdout_sets: dict[str, list[TrainingExample]] = {}
    if tcfg.grok_tracking:
        splitter = StratifiedHoldoutSplitter(
            holdout_frac=args.holdout_frac,
            endgame_threshold=6,
        )
        splits = splitter.split(all_examples)
        train = splits.train
        holdout_sets = {
            "random": splits.random,
            "game_level": splits.game_level,
            "opening_family": splits.opening_family,
            "piece_count": splits.piece_count,
        }
        holdout_sets = {k: v for k, v in holdout_sets.items() if v}
        holdout = splits.random  # Primary holdout for phase gate
        log.info(
            "train=%d  holdout splits: %s",
            len(train),
            ", ".join(f"{k}={len(v)}" for k, v in holdout_sets.items()),
        )
    else:
        holdout_n = max(1, int(len(all_examples) * args.holdout_frac))
        holdout = all_examples[:holdout_n]
        train = all_examples[holdout_n:]
        holdout_sets = {"random": holdout}
        log.info("train=%d  holdout=%d", len(train), holdout_n)

    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        use_harmony_dream=tcfg.use_harmony_dream,
        harmony_ema_decay=tcfg.harmony_ema_decay,
    )

    # --- Grokfast filter (opt-in) ---
    grokfast_filter: GrokfastFilter | None = None
    if tcfg.grokfast:
        grokfast_filter = GrokfastFilter(
            alpha=tcfg.grokfast_alpha,
            lamb=tcfg.grokfast_lamb,
        )
        log.info("grokfast enabled  alpha=%.3f  lamb=%.1f", tcfg.grokfast_alpha, tcfg.grokfast_lamb)

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
        grokfast_filter=grokfast_filter,
    )

    # --- Grok tracker (opt-in) ---
    grok_tracker: GrokTracker | None = None

    monitor = ResourceMonitor()

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        if tcfg.grok_tracking:
            grok_tracker = GrokTracker(
                encoder=encoder,
                backbone=backbone,
                policy_head=policy_head,
                value_head=value_head,
                erank_freq=tcfg.grok_erank_freq,
                spectral_freq=tcfg.grok_spectral_freq,
                onset_threshold=tcfg.grok_onset_threshold,
                on_state_transition=logger.log_grok_state_transition,
            )
            log.info("grok tracking enabled  erank_freq=%d  spectral_freq=%d", tcfg.grok_erank_freq, tcfg.grok_spectral_freq)
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
            monitor.reset()
            step_losses: list[float] = []
            step_grad_norms: list[float] = []
            overflow_count = 0
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
                disable=not use_tqdm,
            )
            data_start = time.monotonic()
            for boards_batch, policies_batch, values_batch in pbar:
                data_time += time.monotonic() - data_start

                compute_start = time.monotonic()
                loss, breakdown = trainer.train_step_tensors(
                    boards_batch, policies_batch, values_batch
                )
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                if grok_tracker is not None:
                    grok_metrics = grok_tracker.step(
                        global_step, breakdown, breakdown.get("grad_norm", 0.0)
                    )
                    logger.log_grok_metrics(global_step, grok_metrics)
                step_losses.append(loss)
                if breakdown.get("overflow", False):
                    overflow_count += 1
                else:
                    step_grad_norms.append(breakdown.get("grad_norm", 0.0))
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    policy=f"{breakdown['policy']:.4f}",
                    value=f"{breakdown['value']:.4f}",
                )
                data_start = time.monotonic()
            pbar.close()
            trainer.scheduler_step()

            epoch_duration = time.monotonic() - epoch_start
            samples_per_sec = len(train) / epoch_duration
            avg_loss = epoch_loss / max(num_batches, 1)
            top1, top5 = measure_accuracy(trainer, holdout, device)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            head_lr = trainer.optimizer.param_groups[2]["lr"]

            logger.log_epoch(epoch, avg_loss, top1, top5, current_lr)
            logger._writer.add_scalar("lr/encoder", current_lr, epoch)
            logger._writer.add_scalar("lr/head", head_lr, epoch)
            logger.log_epoch_timing(epoch, epoch_duration, samples_per_sec)

            resource_metrics = monitor.summarize()
            logger.log_resource_metrics(epoch, resource_metrics)
            logger.log_training_dynamics(epoch, step_losses, step_grad_norms)
            logger.log_pipeline_timing(epoch, data_time, compute_time)

            # --- Grokking detection: evaluate all holdout splits ---
            if grok_tracker is not None:
                holdout_results: dict[str, tuple[float, float]] = {}
                for split_name, split_examples in holdout_sets.items():
                    if split_examples:
                        split_top1, _ = measure_accuracy(
                            trainer, split_examples, device
                        )
                        holdout_results[split_name] = (split_top1, avg_loss)
                grok_epoch_metrics = grok_tracker.epoch(
                    epoch, avg_loss, holdout_results
                )
                logger.log_grok_metrics(epoch, grok_epoch_metrics)

            # --- Consolidated epoch summary via logging ---
            total_time = data_time + compute_time
            summary: dict[str, str] = {
                "epoch": f"{epoch+1}/{args.epochs}",
                "loss": f"{avg_loss:.4f}",
                "policy_loss": f"{breakdown['policy']:.4f}",
                "value_loss": f"{breakdown['value']:.4f}",
                "top1": f"{top1:.1%}",
                "top5": f"{top5:.1%}",
                "lr_enc": f"{current_lr:.2e}",
                "lr_head": f"{head_lr:.2e}",
                "grad_norm_avg": f"{sum(step_grad_norms)/len(step_grad_norms):.3f}" if step_grad_norms else "n/a",
                "grad_norm_peak": f"{max(step_grad_norms):.3f}" if step_grad_norms else "n/a",
                "samples/s": f"{samples_per_sec:.0f}",
                "epoch_time": f"{epoch_duration:.1f}s",
                "data_pct": f"{data_time / total_time:.0%}" if total_time > 0 else "0%",
                "overflows": str(overflow_count),
            }
            if "cpu_percent_avg" in resource_metrics:
                summary["cpu"] = f"{resource_metrics['cpu_percent_avg']:.0f}%/{resource_metrics['cpu_percent_peak']:.0f}%"
            if "ram_mb_avg" in resource_metrics:
                summary["ram"] = f"{resource_metrics['ram_mb_avg']:.0f}mb"
            if "gpu_util_avg" in resource_metrics:
                summary["gpu"] = f"{resource_metrics['gpu_util_avg']:.0f}%/{resource_metrics['gpu_util_peak']:.0f}%"
            if "gpu_temp_avg" in resource_metrics:
                summary["gpu_temp"] = f"{resource_metrics['gpu_temp_avg']:.0f}C"
            if "gpu_power_avg" in resource_metrics:
                summary["gpu_power"] = f"{resource_metrics['gpu_power_avg']:.0f}W"
            logger.log_epoch_summary(summary)

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
                log.info(
                    "PHASE 1 GATE PASSED: top-1 accuracy %s > %s — ready for Phase 2",
                    f"{top1:.1%}", f"{tcfg.phase1_gate:.0%}",
                )
                break

    if grok_tracker is not None:
        grok_tracker.close()

    log.info("best_top1=%s", f"{best_acc:.1%}")


if __name__ == "__main__":
    main()
