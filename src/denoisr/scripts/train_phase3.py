"""Phase 3: RL self-play with MCTS-to-diffusion transition.

Phase 3a: MCTS bootstrap — self-play generates training data.
Phase 3b: Alpha mixing — gradually transition from MCTS to diffusion.

Uses MuZero Reanalyse for sample efficiency: re-run MCTS on old
trajectories with the latest model weights.
"""

import argparse
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
    save_checkpoint,
)
from denoisr.training.phase_orchestrator import PhaseConfig, PhaseOrchestrator
from denoisr.training.reanalyse import ReanalyseActor
from denoisr.training.replay_buffer import PriorityReplayBuffer
from denoisr.training.self_play import (
    SelfPlayActor,
    SelfPlayConfig,
    TemperatureSchedule,
)


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
    parser.add_argument("--output", type=str, default="outputs/phase3.pt")
    parser.add_argument("--save-every", type=int, default=10)
    add_model_args(parser)
    add_training_args(parser)
    add_phase3_args(parser)
    args = parser.parse_args()

    device = detect_device()
    tcfg = full_training_config_from_args(args)
    print(f"Device: {device}")

    # --- Load Phase 2 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    print(f"Loaded Phase 2 checkpoint: d_s={cfg.d_s}")

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

    buffer = PriorityReplayBuffer(capacity=args.buffer_capacity)
    orchestrator = PhaseOrchestrator(
        PhaseConfig(
            phase1_gate=tcfg.phase1_gate,
            phase2_gate=tcfg.phase2_gate,
            alpha_generations=args.alpha_generations,
        )
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
            range(args.games_per_gen),
            desc=f"Gen {gen+1} self-play",
            leave=False,
            unit="game",
            smoothing=0.1,
        )
        for _ in sp_pbar:
            record = actor.play_game(generation=gen)
            buffer.add(record, priority=1.0)
            if record.result == 1.0:
                results["wins"] += 1
            elif record.result == -1.0:
                results["losses"] += 1
            else:
                results["draws"] += 1
            sp_pbar.set_postfix(
                W=results["wins"], D=results["draws"], L=results["losses"]
            )
        sp_pbar.close()

        # 2. Reanalyse old games
        reanalyse_count = 0
        if len(buffer) >= args.reanalyse_per_gen:
            old_records = buffer.sample(args.reanalyse_per_gen)
            ra_pbar = tqdm(
                old_records,
                desc=f"Gen {gen+1} reanalyse",
                leave=False,
                unit="game",
                smoothing=0.1,
            )
            for old_record in ra_pbar:
                examples = reanalyser.reanalyse(old_record)
                reanalyse_count += len(examples)
                ra_pbar.set_postfix(examples=reanalyse_count)
                # TODO: train on reanalysed examples
            ra_pbar.close()

        # 3. Train on replay buffer batch
        # TODO: convert game records to training examples and run trainer

        gen_pbar.set_postfix(
            buf=len(buffer),
            alpha=f"{alpha:.2f}",
            W=results["wins"],
            D=results["draws"],
            L=results["losses"],
        )
        tqdm.write(
            f"Gen {gen+1}/{args.generations}: "
            f"buffer={len(buffer)} alpha={alpha:.2f} "
            f"temp={temp_base:.3f} "
            f"W/D/L={results['wins']}/{results['draws']}/{results['losses']} "
            f"reanalysed={reanalyse_count}"
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
                generation=gen + 1,
            )

    print("Phase 3 training complete.")


if __name__ == "__main__":
    main()
