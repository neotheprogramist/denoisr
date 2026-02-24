"""Pipeline step functions for each training stage.

Each step is idempotent: it checks whether its work has already been done
(via filesystem state or PipelineState) and skips if so.  Heavy imports
(torch, scripts) are deferred to function bodies so that importing this
module stays fast.

All functions accept ``(cfg: PipelineConfig, state: PipelineState, ...)``
and mutate *state* in place to record progress.
"""

import logging
import os
import subprocess
import shutil
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState

log = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _update_state(
    state: PipelineState,
    *,
    phase: str | None = None,
    last_checkpoint: Path | None = None,
    last_data: Path | None = None,
) -> None:
    if phase is not None:
        state.phase = phase
    if last_checkpoint is not None:
        state.last_checkpoint = str(last_checkpoint)
    if last_data is not None:
        state.last_data = str(last_data)
    state.updated_at = _now_iso()


def _run_python_module(module: str, args: list[str]) -> None:
    """Run a project Python module in a subprocess."""
    cmd = [sys.executable, "-m", module, *args]
    log.info("Running: %s", " ".join(cmd))
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        log.warning("Interrupted while running module: %s", module)
        raise
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (130, -signal.SIGINT):
            log.warning("Interrupted module exited with SIGINT: %s", module)
            raise KeyboardInterrupt from None
        raise


# ---------------------------------------------------------------------------
# Step 1: Fetch PGN data
# ---------------------------------------------------------------------------


def step_fetch_data(cfg: PipelineConfig, state: PipelineState) -> None:
    """Download the PGN file if it does not already exist on disk.

    Uses ``wget`` to fetch ``cfg.data.pgn_url`` into ``cfg.data.pgn_path``.
    Skips the download when the target file is already present.
    """
    pgn_path = Path(cfg.data.pgn_path)
    if pgn_path.exists():
        log.info("PGN already exists at %s, skipping download", pgn_path)
        _update_state(state, phase="fetched")
        return

    log.info("Downloading PGN from %s to %s", cfg.data.pgn_url, pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["wget", "-q", "-O", str(pgn_path), cfg.data.pgn_url],
            check=True,
        )
    except KeyboardInterrupt:
        log.warning("Interrupted while downloading PGN to %s", pgn_path)
        raise
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (130, -signal.SIGINT):
            log.warning("PGN download interrupted for %s", pgn_path)
            raise KeyboardInterrupt from None
        raise
    log.info("Download complete: %s", pgn_path)
    _update_state(state, phase="fetched")


# ---------------------------------------------------------------------------
# Step 2: Initialize random model checkpoint
# ---------------------------------------------------------------------------


def step_init_model(cfg: PipelineConfig, state: PipelineState) -> None:
    """Create a random (untrained) model checkpoint.

    Uses the model section of PipelineConfig to build encoder, backbone,
    heads, world model, diffusion module, and consistency projector, then
    saves them all into a single checkpoint file.

    Skips when ``state.last_checkpoint`` already points to an existing file.
    """
    from denoisr.scripts.config import (
        ModelConfig,
        build_backbone,
        build_consistency,
        build_diffusion,
        build_encoder,
        build_policy_head,
        build_value_head,
        build_world_model,
        save_checkpoint,
    )

    if state.last_checkpoint:
        ckpt = Path(state.last_checkpoint)
        if ckpt.exists():
            log.info("Checkpoint already exists at %s, skipping init", ckpt)
            _update_state(
                state,
                phase="model_initialized",
                last_checkpoint=ckpt,
            )
            return

    model_cfg = ModelConfig(
        d_s=cfg.model.d_s,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        ffn_dim=cfg.model.ffn_dim,
        num_timesteps=cfg.model.num_timesteps,
    )

    log.info(
        "Initializing random model: d_s=%d, layers=%d, timesteps=%d",
        model_cfg.d_s,
        model_cfg.num_layers,
        model_cfg.num_timesteps,
    )

    encoder = build_encoder(model_cfg)
    backbone = build_backbone(model_cfg)
    policy_head = build_policy_head(model_cfg)
    value_head = build_value_head(model_cfg)
    world_model = build_world_model(model_cfg)
    diffusion = build_diffusion(model_cfg)
    consistency = build_consistency(model_cfg)

    output_dir = Path(cfg.output.dir)
    ckpt_path = output_dir / "init_model.pt"

    save_checkpoint(
        ckpt_path,
        model_cfg,
        encoder=encoder.state_dict(),
        backbone=backbone.state_dict(),
        policy_head=policy_head.state_dict(),
        value_head=value_head.state_dict(),
        world_model=world_model.state_dict(),
        diffusion=diffusion.state_dict(),
        consistency=consistency.state_dict(),
    )

    _update_state(
        state,
        phase="model_initialized",
        last_checkpoint=ckpt_path,
    )
    log.info("Random model checkpoint saved to %s", ckpt_path)


# ---------------------------------------------------------------------------
# Step 3: Generate training data
# ---------------------------------------------------------------------------


def step_generate_data(
    cfg: PipelineConfig,
    state: PipelineState,
) -> None:
    """Generate Stockfish-evaluated training data from the raw PGN.

    Calls ``generate_to_file()`` with random sampling from the PGN.
    Skips when the output ``.pt`` file already exists.
    """
    from denoisr.scripts.generate_data import generate_to_file

    output_dir = Path(cfg.output.dir)
    data_path = output_dir / "training_data.pt"

    if data_path.exists():
        log.info("Training data already exists at %s, skipping", data_path)
        _update_state(state, last_data=data_path)
        return

    stockfish_cfg = cfg.data.stockfish_path.strip()
    if stockfish_cfg:
        stockfish_path = stockfish_cfg
        stockfish_resolved = shutil.which(stockfish_path)
        if stockfish_resolved is not None:
            stockfish_path = stockfish_resolved
        elif not (Path(stockfish_path).exists() and os.access(stockfish_path, os.X_OK)):
            raise FileNotFoundError(
                "Configured Stockfish binary is not executable: "
                f"{stockfish_path}. Set [data].stockfish_path to a valid "
                "path or leave it empty to auto-detect from PATH."
            )
    else:
        detected = shutil.which("stockfish")
        if not detected:
            raise FileNotFoundError(
                "Stockfish not found in PATH. Set DENOISR_STOCKFISH_PATH "
                "or [data].stockfish_path to an absolute Stockfish binary path."
            )
        stockfish_path = detected

    from denoisr.scripts.config import resolve_workers

    workers = resolve_workers(cfg.data.workers)

    log.info(
        "Generating training data (max_examples=%d, workers=%d, chunksize=%d, chunk_examples=%d)",
        cfg.data.max_examples,
        workers,
        cfg.data.chunksize,
        cfg.data.chunk_examples,
    )

    count = generate_to_file(
        pgn_path=Path(cfg.data.pgn_path),
        output_path=data_path,
        stockfish_path=stockfish_path,
        stockfish_depth=cfg.data.stockfish_depth,
        max_examples=cfg.data.max_examples,
        num_workers=workers,
        chunksize=cfg.data.chunksize,
        chunk_examples=cfg.data.chunk_examples,
    )

    _update_state(state, last_data=data_path)
    log.info("Generated %d examples at %s", count, data_path)


# ---------------------------------------------------------------------------
# Step 4: Train Phase 1
# ---------------------------------------------------------------------------


def step_train_phase1(
    cfg: PipelineConfig,
    state: PipelineState,
) -> None:
    """Train Phase 1 (supervised policy + value) and persist checkpoint."""
    output_dir = Path(cfg.output.dir)
    output_ckpt = output_dir / "phase1.pt"
    if output_ckpt.exists():
        log.info("Phase 1 checkpoint already exists at %s, skipping", output_ckpt)
        _update_state(
            state,
            phase="phase1_complete",
            last_checkpoint=output_ckpt,
        )
        return

    if not state.last_checkpoint:
        raise ValueError(
            "Phase 1 requires an initialized checkpoint in state.last_checkpoint"
        )
    init_ckpt = Path(state.last_checkpoint)
    if not init_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 input checkpoint not found: {init_ckpt}")

    if not state.last_data:
        raise ValueError("Phase 1 requires generated data in state.last_data")
    data_path = Path(state.last_data)
    if not data_path.exists():
        raise FileNotFoundError(f"Phase 1 data file not found: {data_path}")

    args = [
        "--checkpoint",
        str(init_ckpt),
        "--data",
        str(data_path),
        "--holdout-frac",
        str(cfg.phase1.holdout_frac),
        "--epochs",
        str(cfg.phase1.epochs),
        "--lr",
        str(cfg.phase1.lr),
        "--batch-size",
        str(cfg.phase1.batch_size),
        "--warmup-epochs",
        str(cfg.phase1.warmup_epochs),
        "--weight-decay",
        str(cfg.phase1.weight_decay),
        "--tqdm",
        "--output",
        str(output_ckpt),
    ]
    if cfg.output.run_name:
        args.extend(["--run-name", cfg.output.run_name])

    _run_python_module("denoisr.scripts.train_phase1", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 did not produce checkpoint: {output_ckpt}")

    _update_state(
        state,
        phase="phase1_complete",
        last_checkpoint=output_ckpt,
    )
    log.info("Phase 1 complete: %s", output_ckpt)


# ---------------------------------------------------------------------------
# Step 5: Train Phase 2
# ---------------------------------------------------------------------------


def step_train_phase2(cfg: PipelineConfig, state: PipelineState) -> None:
    """Train Phase 2 (world model + diffusion) and persist checkpoint."""
    output_dir = Path(cfg.output.dir)
    output_ckpt = output_dir / "phase2.pt"
    if output_ckpt.exists():
        log.info("Phase 2 checkpoint already exists at %s, skipping", output_ckpt)
        _update_state(
            state,
            phase="phase2_complete",
            last_checkpoint=output_ckpt,
        )
        return

    if not state.last_checkpoint:
        raise ValueError("Phase 2 requires Phase 1 checkpoint in state.last_checkpoint")
    phase1_ckpt = Path(state.last_checkpoint)
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase 2 input checkpoint not found: {phase1_ckpt}")

    pgn_path = Path(cfg.data.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"Phase 2 PGN file not found: {pgn_path}")

    args = [
        "--checkpoint",
        str(phase1_ckpt),
        "--pgn",
        str(pgn_path),
        "--seq-len",
        str(cfg.phase2.seq_len),
        "--max-trajectories",
        str(cfg.phase2.max_trajectories),
        "--batch-size",
        str(cfg.phase2.batch_size),
        "--epochs",
        str(cfg.phase2.epochs),
        "--lr",
        str(cfg.phase2.lr),
        "--tqdm",
        "--output",
        str(output_ckpt),
    ]
    if cfg.output.run_name:
        args.extend(["--run-name", cfg.output.run_name])

    _run_python_module("denoisr.scripts.train_phase2", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 2 did not produce checkpoint: {output_ckpt}")

    _update_state(
        state,
        phase="phase2_complete",
        last_checkpoint=output_ckpt,
    )
    log.info("Phase 2 complete: %s", output_ckpt)


# ---------------------------------------------------------------------------
# Step 6: Train Phase 3
# ---------------------------------------------------------------------------


def step_train_phase3(cfg: PipelineConfig, state: PipelineState) -> None:
    """Train Phase 3 (self-play RL) and persist checkpoint."""
    output_dir = Path(cfg.output.dir)
    output_ckpt = output_dir / "phase3.pt"
    if output_ckpt.exists():
        log.info("Phase 3 checkpoint already exists at %s, skipping", output_ckpt)
        _update_state(
            state,
            phase="phase3_complete",
            last_checkpoint=output_ckpt,
        )
        return

    if not state.last_checkpoint:
        raise ValueError("Phase 3 requires Phase 2 checkpoint in state.last_checkpoint")
    phase2_ckpt = Path(state.last_checkpoint)
    if not phase2_ckpt.exists():
        raise FileNotFoundError(f"Phase 3 input checkpoint not found: {phase2_ckpt}")

    args = [
        "--checkpoint",
        str(phase2_ckpt),
        "--generations",
        str(cfg.phase3.generations),
        "--games-per-gen",
        str(cfg.phase3.games_per_gen),
        "--reanalyse-per-gen",
        str(cfg.phase3.reanalyse_per_gen),
        "--mcts-sims",
        str(cfg.phase3.mcts_sims),
        "--buffer-capacity",
        str(cfg.phase3.buffer_capacity),
        "--alpha-generations",
        str(cfg.phase3.alpha_generations),
        "--lr",
        str(cfg.phase3.lr),
        "--train-batch-size",
        str(cfg.phase3.train_batch_size),
        "--diffusion-steps",
        str(cfg.phase3.diffusion_steps),
        "--aux-updates-per-gen",
        str(cfg.phase3.aux_updates_per_gen),
        "--aux-batch-size",
        str(cfg.phase3.aux_batch_size),
        "--aux-seq-len",
        str(cfg.phase3.aux_seq_len),
        "--self-play-workers",
        str(cfg.phase3.self_play_workers),
        "--reanalyse-workers",
        str(cfg.phase3.reanalyse_workers),
        "--save-every",
        str(cfg.phase3.save_every),
        "--tqdm",
        "--output",
        str(output_ckpt),
    ]
    if cfg.phase3.aux_lr is not None:
        args.extend(["--aux-lr", str(cfg.phase3.aux_lr)])

    _run_python_module("denoisr.scripts.train_phase3", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 3 did not produce checkpoint: {output_ckpt}")

    _update_state(
        state,
        phase="phase3_complete",
        last_checkpoint=output_ckpt,
    )
    log.info("Phase 3 complete: %s", output_ckpt)
