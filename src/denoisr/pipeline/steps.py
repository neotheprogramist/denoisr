"""Pipeline step functions for each training stage.

Each step is idempotent: it checks whether its work has already been done
(via filesystem state or PipelineState) and skips if so.  Heavy imports
(torch, scripts) are deferred to function bodies so that importing this
module stays fast.

All functions accept ``(cfg: PipelineConfig, state: PipelineState, ...)``
and mutate *state* in place to record progress.
"""

import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState

log = logging.getLogger(__name__)


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
        state.phase = "fetched"
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return

    import subprocess

    log.info("Downloading PGN from %s to %s", cfg.data.pgn_url, pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["wget", "-q", "-O", str(pgn_path), cfg.data.pgn_url],
        check=True,
    )
    log.info("Download complete: %s", pgn_path)
    state.phase = "fetched"
    state.updated_at = datetime.now(timezone.utc).isoformat()


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

    state.last_checkpoint = str(ckpt_path)
    state.phase = "model_initialized"
    state.updated_at = datetime.now(timezone.utc).isoformat()
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
        state.last_data = str(data_path)
        state.updated_at = datetime.now(timezone.utc).isoformat()
        return

    stockfish_path = cfg.data.stockfish_path or shutil.which("stockfish") or ""

    from denoisr.scripts.config import resolve_workers

    workers = resolve_workers(cfg.data.workers)

    log.info(
        (
            "Generating training data "
            "(max_examples=%d, workers=%d, chunk_examples=%s)"
        ),
        cfg.data.max_examples,
        workers,
        (
            str(cfg.data.chunk_examples)
            if cfg.data.chunk_examples > 0
            else "single-file"
        ),
    )

    count = generate_to_file(
        pgn_path=Path(cfg.data.pgn_path),
        output_path=data_path,
        stockfish_path=stockfish_path,
        stockfish_depth=cfg.data.stockfish_depth,
        max_examples=cfg.data.max_examples,
        num_workers=workers,
        scratch_dir=(Path(cfg.data.scratch_dir) if cfg.data.scratch_dir else None),
        chunk_examples=cfg.data.chunk_examples,
    )

    state.last_data = str(data_path)
    state.updated_at = datetime.now(timezone.utc).isoformat()
    log.info("Generated %d examples at %s", count, data_path)




# ---------------------------------------------------------------------------
# Step 4: Train Phase 1 (PLACEHOLDER)
# ---------------------------------------------------------------------------


def step_train_phase1(
    cfg: PipelineConfig,
    state: PipelineState,
) -> None:
    """Train Phase 1 (supervised policy + value).

    PLACEHOLDER: logs the parameters and updates state without
    performing actual training.  Will be replaced with the real
    supervised training loop.
    """
    log.info(
        "PLACEHOLDER: Phase 1 training (lr=%s, batch_size=%d)",
        cfg.phase1.lr,
        cfg.phase1.batch_size,
    )

    state.phase = "phase1_complete"
    state.updated_at = datetime.now(timezone.utc).isoformat()

    log.info("PLACEHOLDER: Phase 1 complete")


# ---------------------------------------------------------------------------
# Step 5: Train Phase 2 (PLACEHOLDER)
# ---------------------------------------------------------------------------


def step_train_phase2(cfg: PipelineConfig, state: PipelineState) -> None:
    """Train Phase 2 (diffusion world model).

    PLACEHOLDER: logs the phase parameters and updates state without
    performing actual training.
    """
    log.info(
        "PLACEHOLDER: Phase 2 training (epochs=%d, lr=%s, batch_size=%d, seq_len=%d)",
        cfg.phase2.epochs,
        cfg.phase2.lr,
        cfg.phase2.batch_size,
        cfg.phase2.seq_len,
    )

    state.phase = "phase2_complete"
    state.updated_at = datetime.now(timezone.utc).isoformat()
    log.info("PLACEHOLDER: Phase 2 complete")


# ---------------------------------------------------------------------------
# Step 6: Train Phase 3 (PLACEHOLDER)
# ---------------------------------------------------------------------------


def step_train_phase3(cfg: PipelineConfig, state: PipelineState) -> None:
    """Train Phase 3 (self-play reinforcement learning with MCTS).

    PLACEHOLDER: logs the phase parameters and updates state without
    performing actual training.
    """
    log.info(
        "PLACEHOLDER: Phase 3 training (generations=%d, games/gen=%d, mcts_sims=%d)",
        cfg.phase3.generations,
        cfg.phase3.games_per_gen,
        cfg.phase3.mcts_sims,
    )

    state.phase = "phase3_complete"
    state.updated_at = datetime.now(timezone.utc).isoformat()
    log.info("PLACEHOLDER: Phase 3 complete")
