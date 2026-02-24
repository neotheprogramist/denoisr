"""Phase 3: RL self-play with MCTS-to-diffusion transition.

Phase 3a: MCTS bootstrap — self-play generates training data.
Phase 3b: Alpha mixing — gradually transition from MCTS to diffusion.

Uses MuZero Reanalyse for sample efficiency: re-run MCTS on old
trajectories with the latest model weights.
"""

import argparse
import logging
import random
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import chess
import torch
from torch import Tensor
from tqdm import tqdm

from denoisr.game.chess_game import ChessGame
from denoisr.nn.diffusion import DPMSolverPP
from denoisr.scripts.config import (
    add_phase3_args,
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
    full_training_config_from_args,
    load_checkpoint,
    resolve_dataloader_workers,
    save_checkpoint,
)
from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)
from denoisr.training.phase_orchestrator import PhaseConfig, PhaseOrchestrator
from denoisr.training.phase2_trainer import Phase2Trainer, TrajectoryBatch
from denoisr.training.reanalyse import ReanalyseActor
from denoisr.training.replay_buffer import PriorityReplayBuffer
from denoisr.training.self_play import (
    SelfPlayActor,
    SelfPlayConfig,
    TemperatureSchedule,
)
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import GameRecord, TrainingExample

log = logging.getLogger(__name__)


def _resolve_phase3_workers(requested: int, default_workers: int) -> int:
    if requested > 0:
        return max(1, requested)
    return max(1, min(8, default_workers))


def _records_to_trajectory_batch(
    records: list[GameRecord],
    *,
    board_encoder: object,
    seq_len: int,
) -> TrajectoryBatch | None:
    all_boards: list[Tensor] = []
    all_actions_from: list[Tensor] = []
    all_actions_to: list[Tensor] = []
    all_policies: list[Tensor] = []
    all_legal_masks: list[Tensor] = []
    all_values: list[Tensor] = []
    all_rewards: list[Tensor] = []

    encode = getattr(board_encoder, "encode", None)
    if encode is None:
        raise ValueError("board_encoder must provide encode(board) -> BoardTensor")

    for record in records:
        if len(record.actions) < (seq_len - 1):
            continue

        if record.result == 1.0:
            wdl = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32)
        elif record.result == -1.0:
            wdl = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)
        else:
            wdl = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)

        board = chess.Board()
        boards: list[Tensor] = [encode(board).data]
        from_sqs: list[int] = []
        to_sqs: list[int] = []
        legal_masks: list[Tensor] = []
        move_turns: list[bool] = []

        for action in record.actions:
            legal = torch.zeros(64, 64, dtype=torch.float32)
            for legal_move in board.legal_moves:
                legal[legal_move.from_square, legal_move.to_square] = 1.0
            legal_masks.append(legal)

            move_turns.append(board.turn == chess.WHITE)
            from_sqs.append(action.from_square)
            to_sqs.append(action.to_square)
            board.push(
                chess.Move(action.from_square, action.to_square, action.promotion)
            )
            boards.append(encode(board).data)

        for start in range(0, len(boards) - seq_len + 1, seq_len):
            chunk_boards = boards[start : start + seq_len]
            chunk_from = from_sqs[start : start + seq_len - 1]
            chunk_to = to_sqs[start : start + seq_len - 1]
            chunk_legal = legal_masks[start : start + seq_len - 1]

            policies = torch.zeros(seq_len - 1, 64, 64, dtype=torch.float32)
            for j in range(seq_len - 1):
                policies[j, chunk_from[j], chunk_to[j]] = 1.0

            rewards = torch.zeros(seq_len - 1, dtype=torch.float32)
            for j in range(seq_len - 1):
                was_white = move_turns[start + j]
                side_sign = 1.0 if was_white else -1.0
                rewards[j] = record.result * side_sign

            all_boards.append(torch.stack(chunk_boards))
            all_actions_from.append(torch.tensor(chunk_from, dtype=torch.long))
            all_actions_to.append(torch.tensor(chunk_to, dtype=torch.long))
            all_policies.append(policies)
            all_legal_masks.append(torch.stack(chunk_legal))
            all_values.append(wdl)
            all_rewards.append(rewards)

    if not all_boards:
        return None

    return TrajectoryBatch(
        boards=torch.stack(all_boards),
        actions_from=torch.stack(all_actions_from),
        actions_to=torch.stack(all_actions_to),
        policies=torch.stack(all_policies),
        legal_masks=torch.stack(all_legal_masks),
        values=torch.stack(all_values),
        rewards=torch.stack(all_rewards),
    )


def build_cli_parser() -> argparse.ArgumentParser:
    parser = build_parser("Phase 3: RL self-play")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_PHASE3_CHECKPOINT",
        help="Phase 2 checkpoint path",
    )
    add_env_argument(
        parser,
        "--generations",
        env_var="DENOISR_PHASE3_GENERATIONS",
        type=int,
    )
    add_env_argument(
        parser,
        "--games-per-gen",
        env_var="DENOISR_PHASE3_GAMES_PER_GEN",
        type=int,
    )
    add_env_argument(
        parser,
        "--reanalyse-per-gen",
        env_var="DENOISR_PHASE3_REANALYSE_PER_GEN",
        type=int,
    )
    add_env_argument(
        parser,
        "--mcts-sims",
        env_var="DENOISR_PHASE3_MCTS_SIMS",
        type=int,
    )
    add_env_argument(
        parser,
        "--buffer-capacity",
        env_var="DENOISR_PHASE3_BUFFER_CAPACITY",
        type=int,
    )
    add_env_argument(
        parser,
        "--alpha-generations",
        env_var="DENOISR_PHASE3_ALPHA_GENERATIONS",
        type=int,
    )
    add_env_argument(
        parser,
        "--lr",
        env_var="DENOISR_PHASE3_LR",
        type=float,
    )
    add_env_argument(
        parser,
        "--train-batch-size",
        env_var="DENOISR_PHASE3_TRAIN_BATCH_SIZE",
        type=int,
    )
    add_env_argument(
        parser,
        "--diffusion-steps",
        env_var="DENOISR_PHASE3_DIFFUSION_STEPS",
        type=int,
        help="Denoising steps used to form diffusion policy targets during Phase 3",
    )
    add_env_argument(
        parser,
        "--aux-updates-per-gen",
        env_var="DENOISR_PHASE3_AUX_UPDATES_PER_GEN",
        type=int,
        help="Number of auxiliary Phase 2 trajectory updates per generation",
    )
    add_env_argument(
        parser,
        "--aux-batch-size",
        env_var="DENOISR_PHASE3_AUX_BATCH_SIZE",
        type=int,
        help="Batch size for auxiliary trajectory updates",
    )
    add_env_argument(
        parser,
        "--aux-seq-len",
        env_var="DENOISR_PHASE3_AUX_SEQ_LEN",
        type=int,
        help="Trajectory sequence length for auxiliary updates",
    )
    add_env_argument(
        parser,
        "--aux-lr",
        env_var="DENOISR_PHASE3_AUX_LR",
        type=float,
        default=None,
        required=False,
        help="Learning rate for auxiliary trajectory updates (default: --lr)",
    )
    add_env_argument(
        parser,
        "--self-play-workers",
        env_var="DENOISR_PHASE3_SELF_PLAY_WORKERS",
        type=int,
        help="Parallel self-play workers",
    )
    add_env_argument(
        parser,
        "--reanalyse-workers",
        env_var="DENOISR_PHASE3_REANALYSE_WORKERS",
        type=int,
        help="Parallel reanalyse workers",
    )
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_PHASE3_OUTPUT",
        type=str,
    )
    add_env_argument(
        parser,
        "--save-every",
        env_var="DENOISR_PHASE3_SAVE_EVERY",
        type=int,
    )
    add_training_args(parser)
    add_phase3_args(parser)
    return parser


@graceful_main("denoisr-train-phase3", logger=log)
def main() -> None:
    load_env_file()
    parser = build_cli_parser()
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)

    device = detect_device()
    tcfg = full_training_config_from_args(args)
    default_workers = resolve_dataloader_workers(tcfg.workers)
    self_play_workers = _resolve_phase3_workers(args.self_play_workers, default_workers)
    reanalyse_workers = _resolve_phase3_workers(args.reanalyse_workers, default_workers)
    log.info("device=%s", device)
    log.info(
        "phase3 parallel config: self_play_workers=%d reanalyse_workers=%d",
        self_play_workers,
        reanalyse_workers,
    )

    # --- Load Phase 2 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    log.info("checkpoint loaded d_s=%d", cfg.d_s)

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)
    world_model = build_world_model(cfg).to(device)
    diffusion = build_diffusion(cfg).to(device)
    consistency = build_consistency(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])
    world_model.load_state_dict(state["world_model"])
    diffusion.load_state_dict(state["diffusion"])
    consistency.load_state_dict(state["consistency"])
    schedule = build_schedule(cfg).to(device)
    solver = DPMSolverPP(schedule, num_steps=max(1, args.diffusion_steps))

    encoder.eval()
    backbone.eval()
    policy_head.eval()
    value_head.eval()
    world_model.eval()
    diffusion.eval()

    # --- Model closures for MCTS ---
    @torch.no_grad()
    def policy_value_fn(latent: Tensor) -> tuple[Tensor, Tensor]:
        features = backbone(latent.unsqueeze(0))
        policy = policy_head(features).squeeze(0)
        wdl_logits, _ = value_head(features)
        wdl = torch.softmax(wdl_logits, dim=-1)
        return policy, wdl.squeeze(0)

    @torch.no_grad()
    def world_model_fn(
        latent: Tensor, from_sq: int, to_sq: int
    ) -> tuple[Tensor, float]:
        states = latent.unsqueeze(0).unsqueeze(0)
        a_from = torch.tensor([[from_sq]], device=device)
        a_to = torch.tensor([[to_sq]], device=device)
        next_states, rewards = world_model(states, a_from, a_to)
        return next_states[0, 0], rewards[0, 0].item()

    @torch.no_grad()
    def encode_fn(board_tensor: Tensor) -> Tensor:
        if board_tensor.ndim == 3:
            board_tensor = board_tensor.unsqueeze(0)
        out: Tensor = encoder(board_tensor.to(device, non_blocking=True)).squeeze(0)
        return out

    @torch.no_grad()
    def diffusion_policy_fn(latent: Tensor, legal_mask: Tensor) -> Tensor:
        latent_b = latent.unsqueeze(0)
        imagined = solver.sample(diffusion, latent_b.shape, latent_b, device)
        fused = diffusion.fuse(latent_b, imagined)
        logits = policy_head(backbone(fused)).squeeze(0)
        legal_mask = legal_mask.to(device=logits.device, dtype=torch.bool)
        masked_logits = logits.masked_fill(~legal_mask, float("-inf"))
        if legal_mask.any():
            probs = torch.softmax(masked_logits.reshape(-1), dim=0).reshape(64, 64)
            return probs
        return torch.zeros_like(logits)

    # --- Self-play setup ---
    game = ChessGame()
    board_encoder = build_board_encoder(cfg)

    temp_schedule = TemperatureSchedule(
        base=tcfg.temperature_base,
        explore_moves=tcfg.temperature_explore_moves,
        generation_decay=tcfg.temperature_generation_decay,
    )
    sp_config = SelfPlayConfig(
        num_simulations=args.mcts_sims,
        max_moves=tcfg.max_moves,
        temperature=tcfg.temperature_base,
        c_puct=tcfg.c_puct,
        dirichlet_alpha=tcfg.dirichlet_alpha,
        dirichlet_epsilon=tcfg.dirichlet_epsilon,
        temp_schedule=temp_schedule,
    )

    actor = SelfPlayActor(
        policy_value_fn=policy_value_fn,
        world_model_fn=world_model_fn,
        encode_fn=encode_fn,
        diffusion_policy_fn=diffusion_policy_fn,
        game=game,
        board_encoder=board_encoder,
        config=sp_config,
    )

    reanalyser = ReanalyseActor(
        policy_value_fn=policy_value_fn,
        world_model_fn=world_model_fn,
        encode_fn=encode_fn,
        diffusion_policy_fn=diffusion_policy_fn,
        game=game,
        board_encoder=board_encoder,
        num_simulations=tcfg.reanalyse_simulations,
    )
    _thread_local = threading.local()

    def _get_thread_actor() -> SelfPlayActor:
        thread_actor = getattr(_thread_local, "actor", None)
        if thread_actor is None:
            thread_actor = SelfPlayActor(
                policy_value_fn=policy_value_fn,
                world_model_fn=world_model_fn,
                encode_fn=encode_fn,
                diffusion_policy_fn=diffusion_policy_fn,
                game=ChessGame(),
                board_encoder=build_board_encoder(cfg),
                config=sp_config,
            )
            _thread_local.actor = thread_actor
        return thread_actor

    def _get_thread_reanalyser() -> ReanalyseActor:
        thread_reanalyser = getattr(_thread_local, "reanalyser", None)
        if thread_reanalyser is None:
            thread_reanalyser = ReanalyseActor(
                policy_value_fn=policy_value_fn,
                world_model_fn=world_model_fn,
                encode_fn=encode_fn,
                diffusion_policy_fn=diffusion_policy_fn,
                game=ChessGame(),
                board_encoder=build_board_encoder(cfg),
                num_simulations=tcfg.reanalyse_simulations,
            )
            _thread_local.reanalyser = thread_reanalyser
        return thread_reanalyser

    def _play_game_parallel(generation: int, alpha: float) -> GameRecord:
        return _get_thread_actor().play_game(generation=generation, alpha=alpha)

    def _reanalyse_parallel(record: GameRecord, alpha: float) -> list[TrainingExample]:
        return _get_thread_reanalyser().reanalyse(record, alpha=alpha)

    buffer = PriorityReplayBuffer(capacity=args.buffer_capacity)
    orchestrator = PhaseOrchestrator(
        PhaseConfig(
            phase1_gate=tcfg.phase1_gate,
            phase2_gate=tcfg.phase2_gate,
            alpha_generations=args.alpha_generations,
        )
    )
    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        illegal_penalty_weight=tcfg.illegal_penalty_weight,
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
        total_epochs=max(1, args.generations),
        warmup_epochs=tcfg.warmup_epochs,
        max_grad_norm=tcfg.max_grad_norm,
        weight_decay=tcfg.weight_decay,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        min_lr=tcfg.min_lr,
        use_warm_restarts=tcfg.use_warm_restarts,
    )
    aux_lr = args.aux_lr if args.aux_lr is not None else args.lr
    aux_trainer: Phase2Trainer | None = None
    if args.aux_updates_per_gen > 0:
        aux_loss_fn = ChessLossComputer(
            policy_weight=tcfg.policy_weight,
            value_weight=tcfg.value_weight,
            consistency_weight=tcfg.consistency_weight,
            diffusion_weight=tcfg.diffusion_weight,
            reward_weight=tcfg.reward_weight,
            ply_weight=tcfg.ply_weight,
            use_harmony_dream=tcfg.use_harmony_dream,
            harmony_ema_decay=tcfg.harmony_ema_decay,
        )
        aux_trainer = Phase2Trainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            world_model=world_model,
            diffusion=diffusion,
            consistency=consistency,
            schedule=schedule,
            loss_fn=aux_loss_fn,
            lr=aux_lr,
            device=device,
            max_grad_norm=tcfg.max_grad_norm,
            encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
            weight_decay=tcfg.weight_decay,
            curriculum_initial_fraction=tcfg.curriculum_initial_fraction,
            curriculum_growth=tcfg.curriculum_growth,
            freeze_encoder=False,
        )
        log.info(
            "phase3 auxiliary trainer enabled: updates/gen=%d batch_size=%d seq_len=%d lr=%.2e",
            args.aux_updates_per_gen,
            args.aux_batch_size,
            args.aux_seq_len,
            aux_lr,
        )
    # Advance to phase 3 (gates already passed)
    orchestrator.check_gate({"top1_accuracy": 1.0})
    orchestrator.check_gate({"diffusion_improvement_pp": 100.0})

    # --- Training loop ---
    gen_pbar = tqdm(
        range(args.generations), desc="Generations", unit="gen", smoothing=0.1
    )
    for gen in gen_pbar:
        encoder.eval()
        backbone.eval()
        policy_head.eval()
        value_head.eval()
        world_model.eval()
        diffusion.eval()

        alpha = orchestrator.get_alpha(gen)
        temp_base = temp_schedule.get_temperature(0, gen)

        # 1. Self-play
        results = {"wins": 0, "draws": 0, "losses": 0}
        new_records: list[GameRecord] = []
        sp_pbar = tqdm(
            total=args.games_per_gen,
            desc=f"Gen {gen + 1} self-play",
            leave=False,
            unit="game",
            smoothing=0.1,
        )
        if self_play_workers <= 1:
            for _ in range(args.games_per_gen):
                record = actor.play_game(generation=gen, alpha=alpha)
                new_records.append(record)
                buffer.add(record, priority=1.0)
                if record.result == 1.0:
                    results["wins"] += 1
                elif record.result == -1.0:
                    results["losses"] += 1
                else:
                    results["draws"] += 1
                sp_pbar.update(1)
                sp_pbar.set_postfix(
                    W=results["wins"], D=results["draws"], L=results["losses"]
                )
        else:
            with ThreadPoolExecutor(max_workers=self_play_workers) as executor:
                futures = [
                    executor.submit(_play_game_parallel, gen, alpha)
                    for _ in range(args.games_per_gen)
                ]
                for future in as_completed(futures):
                    record = future.result()
                    new_records.append(record)
                    buffer.add(record, priority=1.0)
                    if record.result == 1.0:
                        results["wins"] += 1
                    elif record.result == -1.0:
                        results["losses"] += 1
                    else:
                        results["draws"] += 1
                    sp_pbar.update(1)
                    sp_pbar.set_postfix(
                        W=results["wins"],
                        D=results["draws"],
                        L=results["losses"],
                    )
        sp_pbar.close()

        # 2. Auxiliary trajectory updates (keeps world model + diffusion learning in Phase 3)
        aux_batches = 0
        avg_aux_loss = 0.0
        if aux_trainer is not None:
            trajectory_data = _records_to_trajectory_batch(
                new_records,
                board_encoder=board_encoder,
                seq_len=max(2, args.aux_seq_len),
            )
            if trajectory_data is not None:
                total_traj = int(trajectory_data.boards.shape[0])
                batch_size = max(1, min(args.aux_batch_size, total_traj))
                aux_loss_sum = 0.0
                for _ in range(args.aux_updates_per_gen):
                    idx = torch.randint(0, total_traj, (batch_size,))
                    batch = TrajectoryBatch(
                        boards=trajectory_data.boards[idx].to(
                            device, non_blocking=True
                        ),
                        actions_from=trajectory_data.actions_from[idx].to(
                            device, non_blocking=True
                        ),
                        actions_to=trajectory_data.actions_to[idx].to(
                            device, non_blocking=True
                        ),
                        policies=trajectory_data.policies[idx].to(
                            device, non_blocking=True
                        ),
                        legal_masks=trajectory_data.legal_masks[idx].to(
                            device, non_blocking=True
                        )
                        if trajectory_data.legal_masks is not None
                        else None,
                        values=trajectory_data.values[idx].to(
                            device, non_blocking=True
                        ),
                        rewards=trajectory_data.rewards[idx].to(
                            device, non_blocking=True
                        ),
                    )
                    loss, _ = aux_trainer.train_step(batch)
                    aux_loss_sum += loss
                    aux_batches += 1
                if aux_batches > 0:
                    aux_trainer.advance_curriculum()
                    avg_aux_loss = aux_loss_sum / aux_batches

        # 3. Reanalyse old games
        reanalyse_count = 0
        reanalysed_examples = []
        if len(buffer) >= args.reanalyse_per_gen:
            old_records = buffer.sample(args.reanalyse_per_gen)
            ra_pbar = tqdm(
                total=len(old_records),
                desc=f"Gen {gen + 1} reanalyse",
                leave=False,
                unit="game",
                smoothing=0.1,
            )
            if reanalyse_workers <= 1:
                for old_record in old_records:
                    examples = reanalyser.reanalyse(old_record, alpha=alpha)
                    reanalyse_count += len(examples)
                    reanalysed_examples.extend(examples)
                    ra_pbar.update(1)
                    ra_pbar.set_postfix(examples=reanalyse_count)
            else:
                with ThreadPoolExecutor(max_workers=reanalyse_workers) as executor:
                    futures = [
                        executor.submit(_reanalyse_parallel, old_record, alpha)
                        for old_record in old_records
                    ]
                    for future in as_completed(futures):
                        examples = future.result()
                        reanalyse_count += len(examples)
                        reanalysed_examples.extend(examples)
                        ra_pbar.update(1)
                        ra_pbar.set_postfix(examples=reanalyse_count)
            ra_pbar.close()

        # 4. Train on replay buffer batch
        train_batches = 0
        avg_train_loss = 0.0
        if reanalysed_examples:
            random.shuffle(reanalysed_examples)
            train_loss_sum = 0.0
            for i in range(0, len(reanalysed_examples), args.train_batch_size):
                batch = reanalysed_examples[i : i + args.train_batch_size]
                loss, _ = trainer.train_step(batch)
                train_loss_sum += loss
                train_batches += 1
            trainer.scheduler_step()
            avg_train_loss = train_loss_sum / max(train_batches, 1)

        gen_pbar.set_postfix(
            buf=len(buffer),
            alpha=f"{alpha:.2f}",
            aux=f"{avg_aux_loss:.3f}" if aux_batches > 0 else "n/a",
            train=f"{avg_train_loss:.3f}" if train_batches > 0 else "n/a",
            W=results["wins"],
            D=results["draws"],
            L=results["losses"],
        )
        log.info(
            "gen %d/%d buffer=%d alpha=%.2f temp=%.3f W/D/L=%d/%d/%d aux_batches=%d aux_loss=%.4f reanalysed=%d train_batches=%d train_loss=%.4f",
            gen + 1,
            args.generations,
            len(buffer),
            alpha,
            temp_base,
            results["wins"],
            results["draws"],
            results["losses"],
            aux_batches,
            avg_aux_loss,
            reanalyse_count,
            train_batches,
            avg_train_loss,
        )

        # 5. Checkpoint
        if (gen + 1) % args.save_every == 0:
            extra_state: dict[str, object] = {}
            if aux_trainer is not None:
                extra_state["phase2_optimizer"] = aux_trainer.optimizer.state_dict()
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
                optimizer=trainer.optimizer.state_dict(),
                generation=gen + 1,
                **extra_state,
            )

    log.info("Phase 3 training complete")


if __name__ == "__main__":
    main()
