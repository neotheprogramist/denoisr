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

import torch
from torch import Tensor
from tqdm import tqdm

from denoisr.game.chess_game import ChessGame
from denoisr.scripts.config import (
    add_model_args,
    add_phase3_args,
    add_training_args,
    build_backbone,
    build_board_encoder,
    build_consistency,
    build_diffusion,
    build_encoder,
    build_policy_head,
    build_value_head,
    build_world_model,
    detect_device,
    full_training_config_from_args,
    load_checkpoint,
    resolve_dataloader_workers,
    save_checkpoint,
)
from denoisr.scripts.interrupts import graceful_main
from denoisr.training.phase_orchestrator import PhaseConfig, PhaseOrchestrator
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


@graceful_main("denoisr-train-phase3", logger=log)
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3: RL self-play")
    parser.add_argument(
        "--checkpoint", required=True, help="Phase 2 checkpoint path"
    )
    parser.add_argument("--generations", type=int, default=1000)
    parser.add_argument("--games-per-gen", type=int, default=100)
    parser.add_argument("--reanalyse-per-gen", type=int, default=50)
    parser.add_argument("--mcts-sims", type=int, default=800)
    parser.add_argument("--buffer-capacity", type=int, default=100_000)
    parser.add_argument("--alpha-generations", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument(
        "--self-play-workers",
        type=int,
        default=0,
        help="Parallel self-play workers (0 = auto: min(8, --workers/auto))",
    )
    parser.add_argument(
        "--reanalyse-workers",
        type=int,
        default=0,
        help="Parallel reanalyse workers (0 = auto: min(8, --workers/auto))",
    )
    parser.add_argument("--output", type=str, default="outputs/phase3.pt")
    parser.add_argument("--save-every", type=int, default=10)
    add_model_args(parser)
    add_training_args(parser)
    add_phase3_args(parser)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device = detect_device()
    tcfg = full_training_config_from_args(args)
    default_workers = resolve_dataloader_workers(tcfg.workers)
    self_play_workers = _resolve_phase3_workers(
        args.self_play_workers, default_workers
    )
    reanalyse_workers = _resolve_phase3_workers(
        args.reanalyse_workers, default_workers
    )
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
        out: Tensor = encoder(board_tensor.unsqueeze(0)).squeeze(0)
        return out

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
        game=game,
        board_encoder=board_encoder,
        config=sp_config,
    )

    reanalyser = ReanalyseActor(
        policy_value_fn=policy_value_fn,
        world_model_fn=world_model_fn,
        encode_fn=encode_fn,
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
                game=ChessGame(),
                board_encoder=build_board_encoder(cfg),
                num_simulations=tcfg.reanalyse_simulations,
            )
            _thread_local.reanalyser = thread_reanalyser
        return thread_reanalyser

    def _play_game_parallel(generation: int) -> GameRecord:
        return _get_thread_actor().play_game(generation=generation)

    def _reanalyse_parallel(record: GameRecord) -> list[TrainingExample]:
        return _get_thread_reanalyser().reanalyse(record)

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
    # Advance to phase 3 (gates already passed)
    orchestrator.check_gate({"top1_accuracy": 1.0})
    orchestrator.check_gate({"diffusion_improvement_pp": 100.0})

    # --- Training loop ---
    gen_pbar = tqdm(range(args.generations), desc="Generations", unit="gen", smoothing=0.1)
    for gen in gen_pbar:
        alpha = orchestrator.get_alpha(gen)
        temp_base = temp_schedule.get_temperature(0, gen)

        # 1. Self-play
        results = {"wins": 0, "draws": 0, "losses": 0}
        sp_pbar = tqdm(
            total=args.games_per_gen,
            desc=f"Gen {gen+1} self-play",
            leave=False,
            unit="game",
            smoothing=0.1,
        )
        if self_play_workers <= 1:
            for _ in range(args.games_per_gen):
                record = actor.play_game(generation=gen)
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
                    executor.submit(_play_game_parallel, gen)
                    for _ in range(args.games_per_gen)
                ]
                for future in as_completed(futures):
                    record = future.result()
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

        # 2. Reanalyse old games
        reanalyse_count = 0
        reanalysed_examples = []
        if len(buffer) >= args.reanalyse_per_gen:
            old_records = buffer.sample(args.reanalyse_per_gen)
            ra_pbar = tqdm(
                total=len(old_records),
                desc=f"Gen {gen+1} reanalyse",
                leave=False,
                unit="game",
                smoothing=0.1,
            )
            if reanalyse_workers <= 1:
                for old_record in old_records:
                    examples = reanalyser.reanalyse(old_record)
                    reanalyse_count += len(examples)
                    reanalysed_examples.extend(examples)
                    ra_pbar.update(1)
                    ra_pbar.set_postfix(examples=reanalyse_count)
            else:
                with ThreadPoolExecutor(max_workers=reanalyse_workers) as executor:
                    futures = [
                        executor.submit(_reanalyse_parallel, old_record)
                        for old_record in old_records
                    ]
                    for future in as_completed(futures):
                        examples = future.result()
                        reanalyse_count += len(examples)
                        reanalysed_examples.extend(examples)
                        ra_pbar.update(1)
                        ra_pbar.set_postfix(examples=reanalyse_count)
            ra_pbar.close()

        # 3. Train on replay buffer batch
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
            train=f"{avg_train_loss:.3f}" if train_batches > 0 else "n/a",
            W=results["wins"],
            D=results["draws"],
            L=results["losses"],
        )
        log.info(
            "gen %d/%d buffer=%d alpha=%.2f temp=%.3f W/D/L=%d/%d/%d reanalysed=%d train_batches=%d train_loss=%.4f",
            gen + 1,
            args.generations,
            len(buffer),
            alpha,
            temp_base,
            results["wins"],
            results["draws"],
            results["losses"],
            reanalyse_count,
            train_batches,
            avg_train_loss,
        )

        # 4. Checkpoint
        if (gen + 1) % args.save_every == 0:
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
            )

    log.info("Phase 3 training complete")


if __name__ == "__main__":
    main()
