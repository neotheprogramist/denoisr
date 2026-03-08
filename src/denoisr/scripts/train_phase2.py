"""Phase 2: World model + diffusion bootstrapping.

Loads Phase 1 checkpoint. Trains diffusion module to denoise future latent
trajectories, with the encoder frozen. Also continues supervised training
with all 6 loss terms active.

Gate to Phase 3: diffusion-conditioned accuracy > single-step by >5pp.
"""

import logging
import math
import time
from pathlib import Path

import chess
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.scripts.config import (
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
    resolve_amp_dtype,
    resolve_dataloader_workers,
    save_checkpoint,
    training_config_from_args,
)
from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)
from denoisr.training.ema import ModelEMA
from denoisr.training.logger import TrainingLogger
from denoisr.training.loss import ChessLossComputer
from denoisr.training.plateau_detector import PlateauDetector
from denoisr.training.prefetch import DevicePrefetcher
from denoisr.training.phase2_trainer import (
    Phase2Trainer,
    TrajectoryBatch,
    evaluate_phase2_gate,
)
from denoisr.training.resource_monitor import ResourceMonitor

log = logging.getLogger(__name__)
_DATALOADER_PREFETCH_FACTOR = 2



def _amp_dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


def extract_trajectories(
    pgn_path: Path,
    encoder: ExtendedBoardEncoder,
    seq_len: int,
    max_trajectories: int,
    use_tqdm: bool = False,
) -> TrajectoryBatch:
    """Extract enriched consecutive board-state trajectories from PGN."""
    streamer = SimplePGNStreamer()

    all_boards: list[torch.Tensor] = []
    all_actions_from: list[torch.Tensor] = []
    all_actions_to: list[torch.Tensor] = []
    all_policies: list[torch.Tensor] = []
    all_legal_masks: list[torch.Tensor] = []
    all_values: list[torch.Tensor] = []
    all_rewards: list[torch.Tensor] = []

    pbar = tqdm(
        total=max_trajectories,
        desc="Extracting trajectories",
        unit="traj",
        smoothing=0.3,
        disable=not use_tqdm,
    )

    for record in streamer.stream(pgn_path):
        if len(record.actions) < (seq_len - 1):
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
        legal_masks: list[torch.Tensor] = []
        # Track whose turn it is before each move
        move_turns: list[bool] = []  # True = WHITE moved

        for action in record.actions:
            legal = torch.zeros(64, 64, dtype=torch.float32)
            for legal_move in board.legal_moves:
                legal[legal_move.from_square, legal_move.to_square] = 1.0
            legal_masks.append(legal)

            move_turns.append(board.turn == chess.WHITE)
            from_sqs.append(action.from_square)
            to_sqs.append(action.to_square)
            move = chess.Move(
                action.from_square,
                action.to_square,
                action.promotion,
            )
            board.push(move)
            boards.append(encoder.encode(board).data)

        for start in range(0, len(boards) - seq_len + 1, seq_len):
            chunk_boards = boards[start : start + seq_len]
            chunk_from = from_sqs[start : start + seq_len - 1]
            chunk_to = to_sqs[start : start + seq_len - 1]
            chunk_legal = legal_masks[start : start + seq_len - 1]

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
            all_legal_masks.append(torch.stack(chunk_legal))
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
        legal_masks=torch.stack(all_legal_masks),
        values=torch.stack(all_values),
        rewards=torch.stack(all_rewards),
    )


@graceful_main("denoisr-train-phase2", logger=log)
def main() -> None:
    load_env_file()
    parser = build_parser("Phase 2: World model + diffusion bootstrapping")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_PHASE2_CHECKPOINT",
        help="Phase 1 checkpoint path",
    )
    add_env_argument(
        parser,
        "--pgn",
        env_var="DENOISR_PHASE2_PGN",
        help="PGN file for trajectory extraction",
    )
    add_env_argument(
        parser,
        "--seq-len",
        env_var="DENOISR_PHASE2_SEQ_LEN",
        type=int,
    )
    add_env_argument(
        parser,
        "--max-trajectories",
        env_var="DENOISR_PHASE2_MAX_TRAJECTORIES",
        type=int,
    )
    add_env_argument(
        parser,
        "--batch-size",
        env_var="DENOISR_PHASE2_BATCH_SIZE",
        type=int,
    )
    add_env_argument(
        parser,
        "--micro-batch-size",
        env_var="DENOISR_PHASE2_MICRO_BATCH_SIZE",
        type=int,
        default=None,
        required=False,
        help=(
            "Optional in-step microbatch size for Phase 2 to reduce CUDA memory "
            "pressure (defaults to --batch-size)."
        ),
    )
    add_env_argument(
        parser,
        "--epochs",
        env_var="DENOISR_PHASE2_EPOCHS",
        type=int,
    )
    add_env_argument(
        parser,
        "--lr",
        env_var="DENOISR_PHASE2_LR",
        type=float,
    )
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_PHASE2_OUTPUT",
        type=str,
    )
    add_env_argument(
        parser,
        "--run-name",
        env_var="DENOISR_RUN_NAME",
        type=str,
        default=None,
        required=False,
        help="log run label attached to training log lines (default: timestamp)",
    )
    add_training_args(parser)
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)

    device = detect_device()
    tcfg = training_config_from_args(args)
    use_tqdm = args.tqdm
    log.info("device=%s", device)

    # --- Load Phase 1 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    log.info(
        "checkpoint loaded  d_s=%d  layers=%d",
        cfg.d_s,
        cfg.num_layers,
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

    requested_amp_dtype = resolve_amp_dtype(tcfg)
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
        total_epochs=args.epochs,
        warmup_epochs=tcfg.warmup_epochs,
        max_grad_norm=tcfg.max_grad_norm,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        weight_decay=tcfg.weight_decay,
        min_lr=tcfg.min_lr,
        curriculum_initial_fraction=tcfg.curriculum_initial_fraction,
        curriculum_growth=tcfg.curriculum_growth,
        amp_dtype=requested_amp_dtype,
        microbatch_size=args.micro_batch_size,
    )
    log.info(
        "AMP config: requested=%s active=%s autocast=%s grad_scaler=%s",
        _amp_dtype_name(requested_amp_dtype),
        _amp_dtype_name(trainer.amp_dtype),
        trainer.amp_autocast_enabled,
        trainer.amp_scaler_enabled,
    )

    # --- EMA shadow model (opt-in) ---
    model_ema: ModelEMA | None = None
    if tcfg.ema_decay > 0:
        model_ema = ModelEMA(
            {
                "encoder": encoder,
                "backbone": backbone,
                "policy_head": policy_head,
                "value_head": value_head,
                "world_model": world_model,
                "diffusion": diffusion_mod,
                "consistency": consistency,
            },
            decay=tcfg.ema_decay,
        )
        log.info("EMA enabled  decay=%.4f", tcfg.ema_decay)

    # --- Extract enriched trajectories ---
    board_encoder = build_board_encoder(cfg)
    trajectory_data = extract_trajectories(
        Path(args.pgn),
        board_encoder,
        args.seq_len,
        args.max_trajectories,
        use_tqdm=use_tqdm,
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
        trajectory_data.legal_masks[:n_train],
        trajectory_data.values[:n_train],
        trajectory_data.rewards[:n_train],
    )
    worker_count = resolve_dataloader_workers(tcfg.workers)
    loader_kwargs: dict[str, object] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": (device.type == "cuda"),
    }
    if worker_count > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = _DATALOADER_PREFETCH_FACTOR
    loader = DataLoader(train_dataset, **loader_kwargs)
    log.info(
        "phase2 dataloader config: workers=%d  batch_size=%d  micro_batch_size=%s",
        worker_count,
        args.batch_size,
        args.micro_batch_size if args.micro_batch_size is not None else "auto",
    )

    monitor = ResourceMonitor()
    plateau_detector = PlateauDetector(warmup_epochs=tcfg.warmup_epochs)

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
                "warmup_epochs": tcfg.warmup_epochs,
                "min_lr": tcfg.min_lr,
                "weight_decay": tcfg.weight_decay,
                "encoder_lr_multiplier": tcfg.encoder_lr_multiplier,
                "amp_dtype": _amp_dtype_name(trainer.amp_dtype),
                "workers": worker_count,
                "micro_batch_size": (
                    args.micro_batch_size if args.micro_batch_size is not None else "auto"
                ),
                "policy_weight": tcfg.policy_weight,
                "value_weight": tcfg.value_weight,
                "consistency_weight": tcfg.consistency_weight,
                "diffusion_weight": tcfg.diffusion_weight,
                "reward_weight": tcfg.reward_weight,
                "state_weight": loss_fn.get_coefficients().get("state", 1.0),
                "ply_weight": tcfg.ply_weight,
                "harmony_dream": tcfg.use_harmony_dream,
            },
            {"best_total_loss": float("inf")},
        )

        global_step = 0

        def _save_current_checkpoint() -> None:
            ema_kwargs: dict[str, object] = {}
            if model_ema is not None:
                for name, sd in model_ema.state_dicts().items():
                    ema_kwargs[f"ema_{name}"] = sd
            save_checkpoint(
                Path(args.output),
                cfg,
                encoder=encoder.state_dict(),
                backbone=backbone.state_dict(),
                policy_head=policy_head.state_dict(),
                value_head=value_head.state_dict(),
                world_model=world_model.state_dict(),
                diffusion=diffusion_mod.state_dict(),
                consistency=consistency.state_dict(),
                **ema_kwargs,
            )

        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            log.info("Epoch %d/%d started", epoch_num, args.epochs)
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_grad_norms: list[float] = []
            top1_sum = 0.0
            top5_sum = 0.0
            topk_count = 0
            overflow_count = 0
            loss_sums = {
                "policy": 0.0,
                "value": 0.0,
                "diffusion": 0.0,
                "consistency": 0.0,
                "state": 0.0,
                "reward": 0.0,
            }
            hidden_loss_sums: dict[str, float] = {}
            hidden_loss_counts: dict[str, int] = {}
            data_time = 0.0

            batch_iter: (
                DevicePrefetcher[tuple[torch.Tensor, ...]]
                | DataLoader[tuple[torch.Tensor, ...]]
            )
            if device.type == "cuda":
                batch_iter = DevicePrefetcher(loader, device)
            else:
                batch_iter = loader

            pbar = tqdm(
                batch_iter,
                desc=f"Epoch {epoch_num}/{args.epochs}",
                total=len(loader),
                leave=False,
                smoothing=0.3,
                disable=not use_tqdm,
            )
            data_start = time.monotonic()
            for boards, af, at, pols, legal, vals, rews in pbar:
                data_time += time.monotonic() - data_start

                batch = TrajectoryBatch(
                    boards=boards.to(device, non_blocking=True),
                    actions_from=af.to(device, non_blocking=True),
                    actions_to=at.to(device, non_blocking=True),
                    policies=pols.to(device, non_blocking=True),
                    legal_masks=legal.to(device, non_blocking=True),
                    values=vals.to(device, non_blocking=True),
                    rewards=rews.to(device, non_blocking=True),
                )

                loss, breakdown = trainer.train_step(batch)
                if model_ema is not None:
                    model_ema.update()

                for key in loss_sums:
                    loss_sums[key] += float(breakdown.get(key, 0.0))
                top1 = float(breakdown.get("top1", float("nan")))
                if math.isfinite(top1):
                    top1_sum += top1
                    topk_count += 1
                top5 = float(breakdown.get("top5", float("nan")))
                if math.isfinite(top5):
                    top5_sum += top5
                for name, raw_value in breakdown.items():
                    if name in {
                        "policy",
                        "value",
                        "diffusion",
                        "consistency",
                        "state",
                        "reward",
                        "total",
                        "grad_norm",
                        "top1",
                        "top5",
                    }:
                        continue
                    if isinstance(raw_value, bool):
                        continue
                    value_scalar = float(raw_value)
                    if not math.isfinite(value_scalar):
                        continue
                    hidden_loss_sums[name] = hidden_loss_sums.get(name, 0.0) + value_scalar
                    hidden_loss_counts[name] = hidden_loss_counts.get(name, 0) + 1
                grad_norm = breakdown.get("grad_norm", 0.0)
                if not math.isfinite(grad_norm):
                    overflow_count += 1
                else:
                    step_grad_norms.append(grad_norm)
                if global_step % 100 == 0:
                    monitor.sample()
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                if use_tqdm:
                    pbar.set_postfix(
                        loss=f"{loss:.4f}",
                        policy=f"{breakdown.get('policy', 0):.4f}",
                        value=f"{breakdown.get('value', 0):.4f}",
                        diff=f"{breakdown.get('diffusion', 0):.4f}",
                    )
                data_start = time.monotonic()
            pbar.close()

            trainer.advance_curriculum()
            trainer.scheduler_step()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(train_dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            samples_per_sec = num_samples / max(epoch_duration, 1e-9)
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            avg_breakdown = {k: v / max(num_batches, 1) for k, v in loss_sums.items()}
            avg_top1 = top1_sum / max(topk_count, 1)
            avg_top5 = top5_sum / max(topk_count, 1)
            overflow_frac = overflow_count / max(num_batches, 1)
            avg_hidden_losses = {
                name: hidden_loss_sums[name] / max(hidden_loss_counts.get(name, 0), 1)
                for name in sorted(hidden_loss_sums)
            }
            loss_coefficients = loss_fn.get_coefficients()

            # Build resource dict in the format log_epoch_line expects
            raw_res = monitor.summarize()
            resource_metrics: dict[str, str] | None = None
            if raw_res:
                resource_metrics = {
                    "cpu_pct": f"{raw_res['cpu_percent_avg']:.0f}%",
                    "cpu_max": f"{raw_res['cpu_percent_peak']:.0f}%",
                    "ram_mb": f"{raw_res['ram_mb_avg']:.0f}",
                }
                if "gpu_util_avg" in raw_res:
                    resource_metrics["gpu_util"] = f"{raw_res['gpu_util_avg']:.0f}%"
                if "gpu_mem_mb_avg" in raw_res:
                    resource_metrics["gpu_mem_mb"] = f"{raw_res['gpu_mem_mb_avg']:.0f}"
                if "gpu_temp_avg" in raw_res:
                    resource_metrics["gpu_temp"] = f"{raw_res['gpu_temp_avg']:.0f}"
                if "gpu_power_avg" in raw_res:
                    resource_metrics["gpu_power"] = f"{raw_res['gpu_power_avg']:.0f}"

            logger.log_epoch_line(
                epoch=epoch_num,
                total_epochs=args.epochs,
                losses={
                    "loss": avg_loss,
                    "pol": avg_breakdown["policy"],
                    "val": avg_breakdown["value"],
                    "diff": avg_breakdown["diffusion"],
                    "cons": avg_breakdown["consistency"],
                    "state": avg_breakdown["state"],
                    "rew": avg_breakdown["reward"],
                },
                accuracy={"top1": avg_top1, "top5": avg_top5},
                lr=current_lr,
                grad_norms=step_grad_norms,
                samples_per_sec=samples_per_sec,
                duration_s=epoch_duration,
                resources=resource_metrics,
                data_pct=data_time / max(epoch_duration, 1e-9) * 100,
                overflows=overflow_count,
                phase="phase2",
            )
            amp_scaler_scale = trainer.amp_scaler_scale()
            scaler_state = "off"
            if amp_scaler_scale is not None:
                scaler_state = f"{amp_scaler_scale:.0f}"
            log.info(
                (
                    "Epoch %d/%d summary: loss=%.4f pol=%.4f val=%.4f "
                    "diff=%.4f cons=%.4f state=%.4f rew=%.4f "
                    "top1=%.2f%% top5=%.2f%% lr=%.2e sps=%.0f "
                    "data=%.1f%% ovf=%d(%.3f%%) amp=%s scaler=%s"
                ),
                epoch_num,
                args.epochs,
                avg_loss,
                avg_breakdown["policy"],
                avg_breakdown["value"],
                avg_breakdown["diffusion"],
                avg_breakdown["consistency"],
                avg_breakdown["state"],
                avg_breakdown["reward"],
                avg_top1,
                avg_top5,
                current_lr,
                samples_per_sec,
                data_time / max(epoch_duration, 1e-9) * 100.0,
                overflow_count,
                overflow_frac * 100.0,
                _amp_dtype_name(trainer.amp_dtype),
                scaler_state,
            )
            if avg_hidden_losses:
                hidden_terms = ",".join(
                    (
                        f"{name}={avg_hidden_losses[name]:.4f}"
                        f"@w={loss_coefficients.get(name, 1.0):.4g}"
                    )
                    for name in sorted(avg_hidden_losses)
                )
                coeff_snapshot = ",".join(
                    (
                        f"{name}={loss_coefficients.get(name, 1.0):.4g}"
                        for name in (
                            "policy",
                            "value",
                            "diffusion",
                            "consistency",
                            "state",
                            "reward",
                            *sorted(avg_hidden_losses),
                        )
                    )
                )
                log.info("Epoch %d hidden losses: %s", epoch_num, hidden_terms)
                logger.log_epoch_summary(
                    {
                        "epoch": f"{epoch_num}/{args.epochs}",
                        "hidden_losses": hidden_terms,
                        "coeffs": coeff_snapshot,
                    }
                )

            # Plateau detection
            grad_norm_avg = (
                sum(step_grad_norms) / len(step_grad_norms) if step_grad_norms else 0.0
            )
            plateau_detector.update(
                epoch,
                grad_norm_avg,
                avg_loss,
                trainer.optimizer.param_groups[0]["lr"],
            )

        # --- Phase 2 gate ---
        log.info(
            "Phase 2 gate on holdout (%d samples)...",
            n_holdout,
        )
        holdout_boards = trajectory_data.boards[n_train:, 0].to(device)
        holdout_legal = trajectory_data.legal_masks[n_train:, 0].to(device)
        holdout_from = trajectory_data.actions_from[n_train:, 0].to(device)
        holdout_to = trajectory_data.actions_to[n_train:, 0].to(device)

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
            legal_mask=holdout_legal,
        )
        log.info(
            "Phase 2 gate: single=%.1f%%  diffusion=%.1f%%  "
            "delta=%.1fpp  threshold=%.1fpp",
            single_acc * 100,
            diff_acc * 100,
            delta,
            tcfg.phase2_gate,
        )
        selected_delta = delta
        selected_source = "base"
        if model_ema is not None:
            with model_ema.apply():
                ema_single, ema_diff, ema_delta = evaluate_phase2_gate(
                    encoder=encoder,
                    backbone=backbone,
                    policy_head=policy_head,
                    diffusion=diffusion_mod,
                    schedule=schedule,
                    boards=holdout_boards,
                    target_from=holdout_from,
                    target_to=holdout_to,
                    device=device,
                    legal_mask=holdout_legal,
                )
            log.info(
                "Phase 2 gate (EMA): single=%.1f%%  diffusion=%.1f%%  delta=%.1fpp",
                ema_single * 100,
                ema_diff * 100,
                ema_delta,
            )
            if ema_delta >= selected_delta:
                selected_delta = ema_delta
                selected_source = "ema"
        if selected_source == "ema" and model_ema is not None:
            with model_ema.apply():
                _save_current_checkpoint()
        else:
            _save_current_checkpoint()
        log.info(
            "Saved Phase 2 checkpoint from %s weights (holdout delta=%.1fpp)",
            selected_source,
            selected_delta,
        )
        if selected_delta > tcfg.phase2_gate:
            log.info(
                "Phase 2 gate PASSED (%s delta %.1fpp > %.1fpp)",
                selected_source,
                selected_delta,
                tcfg.phase2_gate,
            )
        else:
            log.warning(
                "Phase 2 gate NOT PASSED (%s delta %.1fpp <= %.1fpp)."
                " Checkpoint saved -- user decides.",
                selected_source,
                selected_delta,
                tcfg.phase2_gate,
            )


if __name__ == "__main__":
    main()
