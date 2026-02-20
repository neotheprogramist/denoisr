"""Phase 2: World model + diffusion bootstrapping.

Loads Phase 1 checkpoint. Trains diffusion module to denoise future latent
trajectories, with the encoder frozen. Also continues supervised training
with all 6 loss terms active.

Gate to Phase 3: diffusion-conditioned accuracy > single-step by >5pp.
"""

import argparse
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
)
from denoisr.training.diffusion_trainer import DiffusionTrainer
from denoisr.training.logger import TrainingLogger


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
    args = parser.parse_args()

    device = detect_device()
    print(f"Device: {device}")

    # --- Load Phase 1 ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    cfg = resolve_gradient_checkpointing(cfg, args, device)
    print(f"Loaded Phase 1 checkpoint: d_s={cfg.d_s}, layers={cfg.num_layers}")

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
    )

    # --- Extract trajectories ---
    board_encoder = build_board_encoder(cfg)
    trajectories = extract_trajectories(
        Path(args.pgn), board_encoder, args.seq_len, args.max_trajectories
    )
    print(f"Extracted {len(trajectories)} trajectories of length {args.seq_len}")

    # --- Train diffusion ---
    bs = args.batch_size
    best_loss = float("inf")

    all_trajectories = torch.stack(trajectories)  # [N, T, C, 8, 8]
    dataset = TensorDataset(all_trajectories)
    loader = DataLoader(
        dataset, batch_size=bs, shuffle=True,
        num_workers=2, pin_memory=(device.type == "cuda"),
        persistent_workers=True,
    )

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": bs,
                "seq_len": args.seq_len,
                "max_trajectories": args.max_trajectories,
                "d_s": cfg.d_s,
                "num_layers": cfg.num_layers,
            },
            {"best_diffusion_loss": float("inf")},
        )

        global_step = 0

        for epoch in range(args.epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start = time.monotonic()

            pbar = tqdm(
                loader,
                desc=f"Epoch {epoch+1}/{args.epochs}",
                leave=False,
                smoothing=0.3,
            )
            for (batch,) in pbar:
                batch = batch.to(device, non_blocking=True)
                loss, breakdown = diff_trainer.train_step(batch)
                logger.log_train_step(global_step, loss, breakdown)
                if global_step % 100 == 0:
                    logger.log_gpu(global_step)
                global_step += 1
                epoch_loss += loss
                num_batches += 1
                pbar.set_postfix(loss=f"{loss:.4f}")
            pbar.close()

            diff_trainer.advance_curriculum()
            epoch_duration = time.monotonic() - epoch_start
            num_samples = len(dataset)
            avg_loss = epoch_loss / max(num_batches, 1)

            logger.log_diffusion(epoch, avg_loss, diff_trainer.current_max_steps)
            logger.log_epoch_timing(epoch, epoch_duration, num_samples / epoch_duration)

            print(
                f"Epoch {epoch+1}/{args.epochs}: "
                f"avg_diffusion_loss={avg_loss:.4f} "
                f"curriculum_steps={diff_trainer.current_max_steps}"
            )

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

        print(f"Best diffusion loss: {best_loss:.4f}")
        print("Evaluate diffusion vs single-step accuracy to check Phase 2 gate (>5pp).")


if __name__ == "__main__":
    main()
