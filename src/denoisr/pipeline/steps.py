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
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState

log = logging.getLogger(__name__)
_ARTIFACT_META_VERSION = 1
_ARTIFACT_META_SUFFIX = ".meta.json"


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
    # Keep pipeline child scripts in agent-friendly mode regardless of shell env.
    env = os.environ.copy()
    env["DENOISR_TQDM"] = "0"
    try:
        subprocess.run(cmd, check=True, env=env)
    except KeyboardInterrupt:
        log.warning("Interrupted while running module: %s", module)
        raise
    except subprocess.CalledProcessError as exc:
        if exc.returncode in (130, -signal.SIGINT):
            log.warning("Interrupted module exited with SIGINT: %s", module)
            raise KeyboardInterrupt from None
        raise


def _artifact_meta_path(artifact: Path) -> Path:
    return artifact.with_suffix(f"{artifact.suffix}{_ARTIFACT_META_SUFFIX}")


def _canonical_json_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_stamp(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path.resolve()),
            "exists": False,
        }
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _load_artifact_meta(artifact: Path) -> dict[str, Any] | None:
    meta_path = _artifact_meta_path(artifact)
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(
            f"Unreadable artifact metadata {meta_path}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid artifact metadata {meta_path}: expected JSON object")
    return raw


def _artifact_matches(artifact: Path, *, stage: str, fingerprint: str) -> bool:
    if not artifact.exists():
        return False
    meta = _load_artifact_meta(artifact)
    if meta is None:
        return False
    return (
        meta.get("meta_version") == _ARTIFACT_META_VERSION
        and meta.get("stage") == stage
        and meta.get("fingerprint") == fingerprint
    )


def _write_artifact_meta(
    artifact: Path,
    *,
    stage: str,
    fingerprint: str,
    inputs: dict[str, Any],
) -> None:
    meta_path = _artifact_meta_path(artifact)
    payload = {
        "meta_version": _ARTIFACT_META_VERSION,
        "stage": stage,
        "fingerprint": fingerprint,
        "created_at": _now_iso(),
        "inputs": inputs,
    }
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _init_model_fingerprint(cfg: PipelineConfig) -> str:
    return _canonical_json_hash(
        {
            "model": {
                "d_s": cfg.model.d_s,
                "num_heads": cfg.model.num_heads,
                "num_layers": cfg.model.num_layers,
                "ffn_dim": cfg.model.ffn_dim,
                "num_timesteps": cfg.model.num_timesteps,
            }
        }
    )


def _training_data_fingerprint(cfg: PipelineConfig, pgn_path: Path, workers: int) -> str:
    return _canonical_json_hash(
        {
            "data": {
                "pgn_url": cfg.data.pgn_url,
                "pgn_path": cfg.data.pgn_path,
                "stockfish_depth": cfg.data.stockfish_depth,
                "max_examples": cfg.data.max_examples,
                "workers": workers,
                "chunksize": cfg.data.chunksize,
                "chunk_examples": cfg.data.chunk_examples,
            },
            "input": _file_stamp(pgn_path),
        }
    )


def _phase1_fingerprint(
    cfg: PipelineConfig,
    *,
    init_ckpt: Path,
    data_path: Path,
) -> str:
    return _canonical_json_hash(
        {
            "phase1": {
                "epochs": cfg.phase1.epochs,
                "lr": cfg.phase1.lr,
                "batch_size": cfg.phase1.batch_size,
                "holdout_frac": cfg.phase1.holdout_frac,
                "warmup_epochs": cfg.phase1.warmup_epochs,
                "weight_decay": cfg.phase1.weight_decay,
            },
            "run_name": cfg.output.run_name,
            "inputs": {
                "checkpoint": _file_stamp(init_ckpt),
                "data": _file_stamp(data_path),
            },
        }
    )


def _phase2_fingerprint(
    cfg: PipelineConfig,
    *,
    phase1_ckpt: Path,
    pgn_path: Path,
) -> str:
    return _canonical_json_hash(
        {
            "phase2": {
                "epochs": cfg.phase2.epochs,
                "lr": cfg.phase2.lr,
                "batch_size": cfg.phase2.batch_size,
                "seq_len": cfg.phase2.seq_len,
                "max_trajectories": cfg.phase2.max_trajectories,
            },
            "run_name": cfg.output.run_name,
            "inputs": {
                "checkpoint": _file_stamp(phase1_ckpt),
                "pgn": _file_stamp(pgn_path),
            },
        }
    )


def _phase3_fingerprint(cfg: PipelineConfig, *, phase2_ckpt: Path) -> str:
    return _canonical_json_hash(
        {
            "phase3": {
                "generations": cfg.phase3.generations,
                "games_per_gen": cfg.phase3.games_per_gen,
                "reanalyse_per_gen": cfg.phase3.reanalyse_per_gen,
                "mcts_sims": cfg.phase3.mcts_sims,
                "buffer_capacity": cfg.phase3.buffer_capacity,
                "alpha_generations": cfg.phase3.alpha_generations,
                "lr": cfg.phase3.lr,
                "train_batch_size": cfg.phase3.train_batch_size,
                "diffusion_steps": cfg.phase3.diffusion_steps,
                "aux_updates_per_gen": cfg.phase3.aux_updates_per_gen,
                "aux_batch_size": cfg.phase3.aux_batch_size,
                "aux_seq_len": cfg.phase3.aux_seq_len,
                "aux_lr": cfg.phase3.aux_lr,
                "self_play_workers": cfg.phase3.self_play_workers,
                "reanalyse_workers": cfg.phase3.reanalyse_workers,
                "save_every": cfg.phase3.save_every,
            },
            "inputs": {"checkpoint": _file_stamp(phase2_ckpt)},
        }
    )


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
    init_fingerprint = _init_model_fingerprint(cfg)
    if _artifact_matches(
        ckpt_path,
        stage="init_model",
        fingerprint=init_fingerprint,
    ):
        log.info("Init checkpoint already exists at %s with matching metadata, skipping", ckpt_path)
        _update_state(
            state,
            phase="model_initialized",
            last_checkpoint=ckpt_path,
        )
        return

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
    _write_artifact_meta(
        ckpt_path,
        stage="init_model",
        fingerprint=init_fingerprint,
        inputs={
            "model": {
                "d_s": model_cfg.d_s,
                "num_heads": model_cfg.num_heads,
                "num_layers": model_cfg.num_layers,
                "ffn_dim": model_cfg.ffn_dim,
                "num_timesteps": model_cfg.num_timesteps,
            }
        },
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

    stockfish_cfg = cfg.data.stockfish_path.strip()
    if not stockfish_cfg:
        raise ValueError(
            "Pipeline requires explicit data.stockfish_path. "
            "Auto-detection from PATH is disabled."
        )
    stockfish_path = stockfish_cfg
    stockfish_resolved = shutil.which(stockfish_path)
    if stockfish_resolved is not None:
        stockfish_path = stockfish_resolved
    elif not (Path(stockfish_path).exists() and os.access(stockfish_path, os.X_OK)):
        raise FileNotFoundError(
            "Configured Stockfish binary is not executable: "
            f"{stockfish_path}. Set data.stockfish_path to an executable path."
        )

    from denoisr.scripts.config import resolve_workers

    workers = resolve_workers(cfg.data.workers)
    data_fingerprint = _training_data_fingerprint(
        cfg,
        pgn_path=Path(cfg.data.pgn_path),
        workers=workers,
    )
    if _artifact_matches(
        data_path,
        stage="generate_data",
        fingerprint=data_fingerprint,
    ):
        log.info(
            "Training data already exists at %s with matching metadata, skipping",
            data_path,
        )
        _update_state(state, last_data=data_path)
        return
    if data_path.exists():
        existing_meta = _load_artifact_meta(data_path)
        if existing_meta is None:
            raise ValueError(
                "Training data artifact exists without metadata: "
                f"{data_path}. Delete the artifact and regenerate."
            )
        log.info(
            "Training data exists at %s but metadata is stale; regenerating",
            data_path,
        )

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
        use_tqdm=False,
    )
    _write_artifact_meta(
        data_path,
        stage="generate_data",
        fingerprint=data_fingerprint,
        inputs={
            "pgn": _file_stamp(Path(cfg.data.pgn_path)),
            "workers": workers,
            "stockfish_depth": cfg.data.stockfish_depth,
            "max_examples": cfg.data.max_examples,
        },
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
    phase1_fingerprint = _phase1_fingerprint(
        cfg,
        init_ckpt=init_ckpt,
        data_path=data_path,
    )
    if _artifact_matches(output_ckpt, stage="phase1", fingerprint=phase1_fingerprint):
        log.info(
            "Phase 1 checkpoint already exists at %s with matching metadata, skipping",
            output_ckpt,
        )
        _update_state(
            state,
            phase="phase1_complete",
            last_checkpoint=output_ckpt,
        )
        return
    if output_ckpt.exists():
        log.info(
            "Phase 1 checkpoint exists at %s but metadata is stale/missing; rerunning",
            output_ckpt,
        )

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
        "--output",
        str(output_ckpt),
    ]
    if cfg.output.run_name:
        args.extend(["--run-name", cfg.output.run_name])

    _run_python_module("denoisr.scripts.train_phase1", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 1 did not produce checkpoint: {output_ckpt}")
    _write_artifact_meta(
        output_ckpt,
        stage="phase1",
        fingerprint=phase1_fingerprint,
        inputs={
            "checkpoint": _file_stamp(init_ckpt),
            "data": _file_stamp(data_path),
        },
    )

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

    if not state.last_checkpoint:
        raise ValueError("Phase 2 requires Phase 1 checkpoint in state.last_checkpoint")
    phase1_ckpt = Path(state.last_checkpoint)
    if not phase1_ckpt.exists():
        raise FileNotFoundError(f"Phase 2 input checkpoint not found: {phase1_ckpt}")

    pgn_path = Path(cfg.data.pgn_path)
    if not pgn_path.exists():
        raise FileNotFoundError(f"Phase 2 PGN file not found: {pgn_path}")
    phase2_fingerprint = _phase2_fingerprint(
        cfg,
        phase1_ckpt=phase1_ckpt,
        pgn_path=pgn_path,
    )
    if _artifact_matches(output_ckpt, stage="phase2", fingerprint=phase2_fingerprint):
        log.info(
            "Phase 2 checkpoint already exists at %s with matching metadata, skipping",
            output_ckpt,
        )
        _update_state(
            state,
            phase="phase2_complete",
            last_checkpoint=output_ckpt,
        )
        return
    if output_ckpt.exists():
        log.info(
            "Phase 2 checkpoint exists at %s but metadata is stale/missing; rerunning",
            output_ckpt,
        )

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
        "--output",
        str(output_ckpt),
    ]
    if cfg.output.run_name:
        args.extend(["--run-name", cfg.output.run_name])

    _run_python_module("denoisr.scripts.train_phase2", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 2 did not produce checkpoint: {output_ckpt}")
    _write_artifact_meta(
        output_ckpt,
        stage="phase2",
        fingerprint=phase2_fingerprint,
        inputs={
            "checkpoint": _file_stamp(phase1_ckpt),
            "pgn": _file_stamp(pgn_path),
        },
    )

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

    if not state.last_checkpoint:
        raise ValueError("Phase 3 requires Phase 2 checkpoint in state.last_checkpoint")
    phase2_ckpt = Path(state.last_checkpoint)
    if not phase2_ckpt.exists():
        raise FileNotFoundError(f"Phase 3 input checkpoint not found: {phase2_ckpt}")
    phase3_fingerprint = _phase3_fingerprint(cfg, phase2_ckpt=phase2_ckpt)
    if _artifact_matches(output_ckpt, stage="phase3", fingerprint=phase3_fingerprint):
        log.info(
            "Phase 3 checkpoint already exists at %s with matching metadata, skipping",
            output_ckpt,
        )
        _update_state(
            state,
            phase="phase3_complete",
            last_checkpoint=output_ckpt,
        )
        return
    if output_ckpt.exists():
        log.info(
            "Phase 3 checkpoint exists at %s but metadata is stale/missing; rerunning",
            output_ckpt,
        )

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
        "--output",
        str(output_ckpt),
    ]
    if cfg.phase3.aux_lr is not None:
        args.extend(["--aux-lr", str(cfg.phase3.aux_lr)])

    _run_python_module("denoisr.scripts.train_phase3", args)
    if not output_ckpt.exists():
        raise FileNotFoundError(f"Phase 3 did not produce checkpoint: {output_ckpt}")
    _write_artifact_meta(
        output_ckpt,
        stage="phase3",
        fingerprint=phase3_fingerprint,
        inputs={"checkpoint": _file_stamp(phase2_ckpt)},
    )

    _update_state(
        state,
        phase="phase3_complete",
        last_checkpoint=output_ckpt,
    )
    log.info("Phase 3 complete: %s", output_ckpt)
