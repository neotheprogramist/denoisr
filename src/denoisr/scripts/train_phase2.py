"""Phase 2: World model + diffusion bootstrapping.

Loads Phase 1 checkpoint. Trains diffusion module to denoise future latent
trajectories, with the encoder frozen. Also continues supervised training
with all 6 loss terms active.

Gate to Phase 3: diffusion-conditioned accuracy > single-step by >5pp.
"""

import argparse
import logging
import math
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
from denoisr.training.logger import TrainingLogger
from denoisr.training.loss import ChessLossComputer
from denoisr.training.plateau_detector import PlateauDetector
from denoisr.training.phase2_trainer import (
    Phase2Trainer,
    TrajectoryBatch,
    evaluate_phase2_gate,
)
from denoisr.training.resource_monitor import ResourceMonitor

log = logging.getLogger(__name__)


def extract_trajectories(
    pgn_path: Path,
    encoder: SimpleBoardEncoder | ExtendedBoardEncoder,
    seq_len: int,
    max_trajectories: int,
    min_elo: int | None = None,
) -> TrajectoryBatch:
    """Extract enriched consecutive board-state trajectories from PGN.

    When min_elo is set, games where min(white_elo, black_elo) < min_elo
    are skipped, ensuring training trajectories come from stronger games.
    """
    streamer = SimplePGNStreamer()

    all_boards: list[torch.Tensor] = []
    all_actions_from: list[torch.Tensor] = []
    all_actions_to: list[torch.Tensor] = []
    all_policies: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []

    pbar = tqdm(
        total=max_trajectories, desc="Extracting trajectories",
        unit="traj", smoothing=0.3,
    )

    for record in streamer.stream(pgn_path):
        # Elo filtering: skip games below threshold
        if min_elo is not None:
            w_elo, b_elo = record.white_elo, record.black_elo
            if w_elo is not None and b_elo is not None:
                if min(w_elo, b_elo) < min_elo:
                    continue
            elif w_elo is not None:
                if w_elo < min_elo:
                    continue
            elif b_elo is not None:
                if b_elo < min_elo:
                    continue
            else:
                # No Elo data at all — skip when filtering is requested
                continue
        if len(record.actions) < seq_len:
            continue

        # WDL from game result
        if record.result == 1.0:
            wdl = torch.tensor([1.0, 0.0, 0.0])
        elif record.result == -1.0:
            wdl = torch.tensor([0.0, 0.0, 1.0])
        else:
            wdl = torch.tensor([0.0, 1.0, 0.0])

        # +1 White wins, -1 Black wins, 0 draw
        result_signal = record.result  # already 1.0/-1.0/0.0

        board = chess.Board()
        boards: list[torch.Tensor] = [encoder.encode(board).data]
        from_sqs: list[int] = []
        to_sqs: list[int] = []
        # Track whose turn it is before each move
        move_turns: list[bool] = []  # True = WHITE moved

        for action in record.actions:
            move_turns.append(board.turn == chess.WHITE)
            from_sqs.append(action.from_square)
            to_sqs.append(action.to_square)
            move = chess.Move(
                action.from_square, action.to_square,
                action.promotion,
            )
            board.push(move)
            boards.append(encoder.encode(board).data)

        for start in range(0, len(boards) - seq_len, seq_len):
            chunk_boards = boards[start : start + seq_len]
            chunk_from = from_sqs[start : start + seq_len - 1]
            chunk_to = to_sqs[start : start + seq_len - 1]

            policies = torch.zeros(seq_len - 1, 64, 64)
            for j in range(seq_len - 1):
                policies[j, chunk_from[j], chunk_to[j]] = 1.0

            rewards = torch.zeros(seq_len - 1)
            for j in range(seq_len - 1):
                # Use actual board turn instead of position index parity
                was_white = move_turns[start + j]
                side_sign = 1.0 if was_white else -1.0
                rewards[j] = result_signal * side_sign

            all_boards.append(torch.stack(chunk_boards))
            all_actions_from.append(
                torch.tensor(chunk_from, dtype=torch.long),
            )
            all_actions_to.append(
                torch.tensor(chunk_to, dtype=torch.long),
            )
            all_policies.append(policies)
            all_values.append(wdl)
            all_rewards.append(rewards)

            pbar.update(1)
            if len(all_boards) >= max_trajectories:
                break
        if len(all_boards) >= max_trajectories:
            break

    pbar.close()
    if not all_boards:
        raise ValueError("No valid trajectories extracted from PGN")

    return TrajectoryBatch(
        boards=torch.stack(all_boards),
        actions_from=torch.stack(all_actions_from),
        actions_to=torch.stack(all_actions_to),
        policies=torch.stack(all_policies),
        values=torch.stack(all_values),
        rewards=torch.stack(all_rewards),
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase 2: World model + diffusion bootstrapping"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Phase 1 checkpoint path"
    )
    parser.add_argument(
        "--pgn", required=True, help="PGN file for trajectory extraction"
    )
    parser.add_argument("--seq-len", type=int, default=10)
    parser.add_argument("--max-trajectories", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--min-elo", type=int, default=1200,
        help="Minimum Elo to include games (min of white/black, default: 1200)",
    )
    parser.add_argument("--output", type=str, default="outputs/phase2.pt")
    parser.add_argument(
        "--run-name", type=str, default=None,
        help="TensorBoard run name (default: timestamp)",
    )
    add_model_args(parser)
    add_training_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = detect_device()
    tcfg = training_config_from_args(args)
    use_tqdm = args.tqdm
    log.info("device=%s", device)

    # --- Load Phase 1 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    log.info(
        "checkpoint loaded  d_s=%d  layers=%d", cfg.d_s, cfg.num_layers,
    )

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
    diffusion_mod = build_diffusion(cfg).to(device)
    consistency = build_consistency(cfg).to(device)
    schedule = build_schedule(cfg).to(device)

    encoder = maybe_compile(encoder, device)
    backbone = maybe_compile(backbone, device)
    diffusion_mod = maybe_compile(diffusion_mod, device)

    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        consistency_weight=tcfg.consistency_weight,
        diffusion_weight=tcfg.diffusion_weight,
        reward_weight=tcfg.reward_weight,
        ply_weight=tcfg.ply_weight,
        use_harmony_dream=tcfg.use_harmony_dream,
        harmony_ema_decay=tcfg.harmony_ema_decay,
    )

    trainer = Phase2Trainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        world_model=world_model,
        diffusion=diffusion_mod,
        consistency=consistency,
        schedule=schedule,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
        max_grad_norm=tcfg.max_grad_norm,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        weight_decay=tcfg.weight_decay,
        curriculum_initial_fraction=tcfg.curriculum_initial_fraction,
        curriculum_growth=tcfg.curriculum_growth,
    )

    # --- Extract enriched trajectories ---
    board_encoder = build_board_encoder(cfg)
    trajectory_data = extract_trajectories(
        Path(args.pgn), board_encoder,
        args.seq_len, args.max_trajectories,
        min_elo=args.min_elo,
    )
    N = trajectory_data.boards.shape[0]
    log.info("trajectories=%d  seq_len=%d", N, args.seq_len)

    # --- Train/holdout split (95/5) ---
    n_holdout = max(1, int(N * 0.05))
    n_train = N - n_holdout

    train_dataset = TensorDataset(
        trajectory_data.boards[:n_train],
        trajectory_data.actions_from[:n_train],
        trajectory_data.actions_to[:n_train],
        trajectory_data.policies[:n_train],
        trajectory_data.values[:n_train],
        trajectory_data.rewards[:n_train],
    )
    loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=tcfg.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    monitor = ResourceMonitor()
    plateau_detector = PlateauDetector()
    best_loss = float("inf")

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "seq_len": args.seq_len,
                "max_trajectories": args.max_trajectories,
                "d_s": cfg.d_s,
                "num_layers": cfg.num_layers,
                "diffusion_layers": cfg.diffusion_layers,
                "num_timesteps": cfg.num_timesteps,
                "max_grad_norm": tcfg.max_grad_norm,
                "harmony_dream": tcfg.use_harmony_dream,
            },
            {"best_total_loss": float("inf")},
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
            last_breakdown: dict[str, float] = {}
            data_time = 0.0
            compute_time = 0.0

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False, smoothing=0.3,
                disable=not use_tqdm,
            )
            data_start = time.monotonic()
            for boards, af, at, pols, vals, rews in pbar:
                data_time += time.monotonic() - data_start

                batch = TrajectoryBatch(
                    boards=boards.to(device, non_blocking=True),
                    actions_from=af.to(device, non_blocking=True),
                    actions_to=at.to(device, non_blocking=True),
                    policies=pols.to(device, non_blocking=True),
                    values=vals.to(device, non_blocking=True),
                    rewards=rews.to(device, non_blocking=True),
                )

                compute_start = time.monotonic()
                loss, breakdown = trainer.train_step(batch)
                compute_time += time.monotonic() - compute_start

                logger.log_train_step(global_step, loss, breakdown)
                last_breakdown = breakdown
                step_losses.append(loss)
                grad_norm = breakdown.get("grad_norm", 0.0)
                if not math.isfinite(grad_norm):
                    overflow_count += 1
                else:
                    step_grad_norms.append(grad_norm)
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    policy=f"{breakdown.get('policy', 0):.4f}",
                    value=f"{breakdown.get('value', 0):.4f}",
                    diff=f"{breakdown.get('diffusion', 0):.4f}",
                )
                data_start = time.monotonic()
            pbar.close()

            trainer.advance_curriculum()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(train_dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            logger.log_diffusion(
                epoch, avg_loss, trainer.current_max_steps,
            )
            logger.log_epoch_timing(
                epoch, epoch_duration,
                num_samples / epoch_duration,
            )

            resource_metrics = monitor.summarize()
            logger.log_resource_metrics(epoch, resource_metrics)
            logger.log_training_dynamics(
                epoch, step_losses, step_grad_norms,
            )
            logger.log_pipeline_timing(
                epoch, data_time, compute_time,
            )

            # LR logging
            logger._writer.add_scalar("lr/backbone", trainer.optimizer.param_groups[0]["lr"], epoch)
            logger._writer.add_scalar("lr/heads", trainer.optimizer.param_groups[1]["lr"], epoch)

            total_time = data_time + compute_time
            summary: dict[str, str] = {
                "epoch": f"{epoch+1}/{args.epochs}",
                "total_loss": f"{avg_loss:.4f}",
                "policy_loss": f"{last_breakdown.get('policy', 0):.4f}",
                "value_loss": f"{last_breakdown.get('value', 0):.4f}",
                "diffusion_loss": f"{last_breakdown.get('diffusion', 0):.4f}",
                "consistency_loss": f"{last_breakdown.get('consistency', 0):.4f}",
                "state_loss": f"{last_breakdown.get('state', 0):.4f}",
                "reward_loss": f"{last_breakdown.get('reward', 0):.4f}",
                "curriculum_steps": str(trainer.current_max_steps),
                "grad_norm_avg": (
                    f"{sum(step_grad_norms)/len(step_grad_norms):.3f}"
                    if step_grad_norms else "n/a"
                ),
                "grad_norm_peak": (
                    f"{max(step_grad_norms):.3f}"
                    if step_grad_norms else "n/a"
                ),
                "overflows": str(overflow_count),
                "samples/s": f"{num_samples / epoch_duration:.0f}",
                "epoch_time": f"{epoch_duration:.1f}s",
                "data_pct": (
                    f"{data_time / total_time:.0%}"
                    if total_time > 0 else "0%"
                ),
            }
            resource_summary = monitor.summarize()
            if "cpu_percent_avg" in resource_summary:
                summary["cpu"] = f"{resource_summary['cpu_percent_avg']:.0f}%/{resource_summary['cpu_percent_peak']:.0f}%"
            if "ram_mb_avg" in resource_summary:
                summary["ram"] = f"{resource_summary['ram_mb_avg']:.0f}mb"
            if "gpu_util_avg" in resource_summary:
                summary["gpu"] = f"{resource_summary['gpu_util_avg']:.0f}%/{resource_summary['gpu_util_peak']:.0f}%"
            if "gpu_temp_avg" in resource_summary:
                summary["gpu_temp"] = f"{resource_summary['gpu_temp_avg']:.0f}C"
            if "gpu_power_avg" in resource_summary:
                summary["gpu_power"] = f"{resource_summary['gpu_power_avg']:.0f}W"
            if tcfg.use_harmony_dream:
                for k, v in loss_fn.get_coefficients().items():
                    summary[f"hd_{k}"] = f"{v:.3f}"
            logger.log_epoch_summary(summary)

            # Plateau detection
            grad_norm_avg = (
                sum(step_grad_norms) / len(step_grad_norms)
                if step_grad_norms else 0.0
            )
            plateau_detector.update(
                epoch, grad_norm_avg, avg_loss,
                trainer.optimizer.param_groups[0]["lr"],
            )

            if avg_loss < best_loss:
                best_loss = avg_loss
                save_checkpoint(
                    Path(args.output), cfg,
                    encoder=encoder.state_dict(),
                    backbone=backbone.state_dict(),
                    policy_head=policy_head.state_dict(),
                    value_head=value_head.state_dict(),
                    world_model=world_model.state_dict(),
                    diffusion=diffusion_mod.state_dict(),
                    consistency=consistency.state_dict(),
                )

        # --- Phase 2 gate ---
        log.info(
            "Phase 2 gate on holdout (%d samples)...", n_holdout,
        )
        holdout_boards = trajectory_data.boards[
            n_train:, 0
        ].to(device)
        holdout_from = trajectory_data.actions_from[
            n_train:, 0
        ].to(device)
        holdout_to = trajectory_data.actions_to[
            n_train:, 0
        ].to(device)

        single_acc, diff_acc, delta = evaluate_phase2_gate(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            diffusion=diffusion_mod,
            schedule=schedule,
            boards=holdout_boards,
            target_from=holdout_from,
            target_to=holdout_to,
            device=device,
        )
        log.info(
            "Phase 2 gate: single=%.1f%%  diffusion=%.1f%%  "
            "delta=%.1fpp  threshold=%.1fpp",
            single_acc * 100, diff_acc * 100,
            delta, tcfg.phase2_gate,
        )
        if delta > tcfg.phase2_gate:
            log.info(
                "Phase 2 gate PASSED (delta %.1fpp > %.1fpp)",
                delta, tcfg.phase2_gate,
            )
        else:
            log.warning(
                "Phase 2 gate NOT PASSED (delta %.1fpp <= %.1fpp)."
                " Checkpoint saved -- user decides.",
                delta, tcfg.phase2_gate,
            )


if __name__ == "__main__":
    main()
