"""Phase 2: World model + diffusion bootstrapping.

Loads Phase 1 checkpoint. Trains diffusion module to denoise future latent
trajectories, with the encoder frozen. Also continues supervised training
with all 6 loss terms active.

Gate to Phase 3: diffusion-conditioned accuracy > single-step by >5pp.
"""

import argparse
import logging
import time
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.scripts.config import (
    add_model_args,
    add_training_args,
    build_backbone,
    build_board_encoder,
    build_consistency,
    build_diffusion,
    build_encoder,
    build_policy_head,
    build_schedule,
    build_value_head,
    build_world_model,
    detect_device,
    load_checkpoint,
    maybe_compile,
    resolve_gradient_checkpointing,
    save_checkpoint,
    training_config_from_args,
)
from denoisr.training.diffusion_trainer import DiffusionTrainer
from denoisr.training.logger import TrainingLogger
from denoisr.training.resource_monitor import ResourceMonitor

log = logging.getLogger(__name__)


def extract_trajectories(
    pgn_path: Path,
    encoder: SimpleBoardEncoder | ExtendedBoardEncoder,
    seq_len: int,
    max_trajectories: int,
) -> list[torch.Tensor]:
    """Extract consecutive board-state trajectories from PGN games."""
    streamer = SimplePGNStreamer()
    trajectories: list[torch.Tensor] = []

    pbar = tqdm(total=max_trajectories, desc="Extracting trajectories", unit="traj", smoothing=0.3)

    for record in streamer.stream(pgn_path):
        if len(record.actions) < seq_len:
            continue

        board = chess.Board()
        boards: list[torch.Tensor] = [encoder.encode(board).data]

        for action in record.actions:
            move = chess.Move(
                action.from_square, action.to_square, action.promotion
            )
            board.push(move)
            boards.append(encoder.encode(board).data)

        for start in range(0, len(boards) - seq_len, seq_len):
            chunk = boards[start : start + seq_len]
            trajectories.append(torch.stack(chunk))
            pbar.update(1)
            if len(trajectories) >= max_trajectories:
                pbar.close()
                return trajectories

    pbar.close()
    return trajectories


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: Diffusion bootstrapping"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Phase 1 checkpoint path"
    )
    parser.add_argument(
        "--pgn", required=True, help="PGN file for trajectory extraction"
    )
    parser.add_argument("--seq-len", type=int, default=5)
    parser.add_argument("--max-trajectories", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output", type=str, default="outputs/phase2.pt")
    parser.add_argument("--run-name", type=str, default=None, help="TensorBoard run name (default: timestamp)")
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

    # --- Load Phase 1 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    log.info("checkpoint loaded  d_s=%d  layers=%d", cfg.d_s, cfg.num_layers)

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])

    # --- Build Phase 2 modules ---
    world_model = build_world_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    consistency = build_consistency(cfg).to(device)
    schedule = build_schedule(cfg).to(device)

    encoder = maybe_compile(encoder, device)
    backbone = maybe_compile(backbone, device)
    diffusion = maybe_compile(diffusion, device)

    diff_trainer = DiffusionTrainer(
        encoder=encoder,
        diffusion=diffusion,
        schedule=schedule,
        lr=args.lr,
        device=device,
        max_grad_norm=tcfg.max_grad_norm,
        curriculum_initial_fraction=tcfg.curriculum_initial_fraction,
        curriculum_growth=tcfg.curriculum_growth,
    )

    # --- Extract trajectories ---
    board_encoder = build_board_encoder(cfg)
    trajectories = extract_trajectories(
        Path(args.pgn), board_encoder, args.seq_len, args.max_trajectories
    )
    log.info("trajectories=%d  seq_len=%d", len(trajectories), args.seq_len)

    # --- Train diffusion ---
    bs = args.batch_size
    best_loss = float("inf")

    all_trajectories = torch.stack(trajectories)  # [N, T, C, 8, 8]
    dataset = TensorDataset(all_trajectories)
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=tcfg.num_workers, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    monitor = ResourceMonitor()

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": bs,
                "epochs": args.epochs,
                "seq_len": args.seq_len,
                "max_trajectories": args.max_trajectories,
                "d_s": cfg.d_s,
                "num_layers": cfg.num_layers,
                "diffusion_layers": cfg.diffusion_layers,
                "num_timesteps": cfg.num_timesteps,
                "max_grad_norm": tcfg.max_grad_norm,
                "curriculum_initial_fraction": tcfg.curriculum_initial_fraction,
                "curriculum_growth": tcfg.curriculum_growth,
                "num_workers": tcfg.num_workers,
            },
            {"best_diffusion_loss": float("inf")},
        )

        global_step = 0

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_losses: list[float] = []
            step_grad_norms: list[float] = []
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
                disable=not use_tqdm,
            )
            data_start = time.monotonic()
            for (batch,) in pbar:
                data_time += time.monotonic() - data_start

                batch = batch.to(device, non_blocking=True)
                compute_start = time.monotonic()
                loss, breakdown = diff_trainer.train_step(batch)
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                step_losses.append(loss)
                step_grad_norms.append(breakdown.get("grad_norm", 0.0))
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
                data_start = time.monotonic()
            pbar.close()

            diff_trainer.advance_curriculum()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            logger.log_diffusion(epoch, avg_loss, diff_trainer.current_max_steps)
            logger.log_epoch_timing(epoch, epoch_duration, num_samples / epoch_duration)

            resource_metrics = monitor.summarize()
            logger.log_resource_metrics(epoch, resource_metrics)
            logger.log_training_dynamics(epoch, step_losses, step_grad_norms)
            logger.log_pipeline_timing(epoch, data_time, compute_time)

            # --- Consolidated epoch summary via logging ---
            total_time = data_time + compute_time
            summary: dict[str, str] = {
                "epoch": f"{epoch+1}/{args.epochs}",
                "diffusion_loss": f"{avg_loss:.4f}",
                "curriculum_steps": str(diff_trainer.current_max_steps),
                "grad_norm_avg": f"{sum(step_grad_norms)/len(step_grad_norms):.3f}" if step_grad_norms else "n/a",
                "grad_norm_peak": f"{max(step_grad_norms):.3f}" if step_grad_norms else "n/a",
                "samples/s": f"{num_samples / epoch_duration:.0f}",
                "epoch_time": f"{epoch_duration:.1f}s",
                "data_pct": f"{data_time / total_time:.0%}" if total_time > 0 else "0%",
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

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    Path(args.output),
                    cfg,
                    encoder=encoder.state_dict(),
                    backbone=backbone.state_dict(),
                    policy_head=policy_head.state_dict(),
                    value_head=value_head.state_dict(),
                    world_model=world_model.state_dict(),
                    diffusion=diffusion.state_dict(),
                    consistency=consistency.state_dict(),
                )

        log.info("best_diffusion_loss=%s", f"{best_loss:.4f}")
        log.info("Evaluate diffusion vs single-step accuracy to check Phase 2 gate (>5pp).")


if __name__ == "__main__":
    main()
