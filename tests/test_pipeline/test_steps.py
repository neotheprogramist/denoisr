"""Tests for pipeline step functions."""

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from denoisr.pipeline.config import (
    DataConfig,
    ModelSectionConfig,
    OutputConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    PipelineConfig,
)
from denoisr.pipeline.state import PipelineState
from denoisr.pipeline.steps import (
    _run_python_module,
    step_fetch_data,
    step_generate_data,
    step_init_model,
    step_train_phase1,
    step_train_phase2,
    step_train_phase3,
)


def _make_cfg(tmp_path: Path) -> PipelineConfig:
    """Build a PipelineConfig pointing at tmp_path for all I/O."""
    return PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            pgn_path=str(tmp_path / "data" / "lichess.pgn.zst"),
            stockfish_path="",
            stockfish_depth=1,
            max_examples=10,
            workers=1,
            chunksize=16,
            chunk_examples=32,
        ),
        model=ModelSectionConfig(
            d_s=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            num_timesteps=10,
        ),
        phase1=Phase1Config(
            epochs=5,
            lr=3e-4,
            batch_size=32,
            holdout_frac=0.05,
            warmup_epochs=1,
            weight_decay=1e-4,
        ),
        phase2=Phase2Config(
            epochs=5,
            lr=3e-4,
            batch_size=16,
            seq_len=4,
            max_trajectories=20,
        ),
        phase3=Phase3Config(
            generations=10,
            games_per_gen=5,
            reanalyse_per_gen=2,
            mcts_sims=50,
            buffer_capacity=128,
            alpha_generations=4,
            lr=1e-4,
            train_batch_size=16,
            diffusion_steps=4,
            aux_updates_per_gen=1,
            aux_batch_size=8,
            aux_seq_len=4,
            aux_lr=1e-4,
            self_play_workers=0,
            reanalyse_workers=0,
            save_every=1,
        ),
        output=OutputConfig(
            dir=str(tmp_path / "outputs"),
            run_name="test-run",
        ),
    )


# -- step_init_model -------------------------------------------------------


def test_init_model_creates_checkpoint(tmp_path: Path) -> None:
    """step_init_model creates a valid checkpoint file and updates state."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    step_init_model(cfg, state)

    ckpt_path = Path(state.last_checkpoint)
    assert ckpt_path.exists(), "Checkpoint file was not created"

    # Verify checkpoint contents are loadable
    data = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    assert "config" in data
    assert "encoder" in data
    assert "backbone" in data
    assert "policy_head" in data
    assert "value_head" in data
    assert "world_model" in data
    assert "diffusion" in data
    assert "consistency" in data

    # Verify config matches what we requested
    assert data["config"]["d_s"] == 64
    assert data["config"]["num_layers"] == 2

    # Verify state was updated
    assert state.phase == "model_initialized"
    assert state.updated_at != ""


def test_init_model_skips_when_checkpoint_exists(tmp_path: Path) -> None:
    """step_init_model does not overwrite an existing checkpoint."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    # Run once to create checkpoint
    step_init_model(cfg, state)
    first_checkpoint = state.last_checkpoint
    first_mtime = Path(first_checkpoint).stat().st_mtime

    # Run again -- should skip
    step_init_model(cfg, state)
    assert state.last_checkpoint == first_checkpoint
    assert Path(first_checkpoint).stat().st_mtime == first_mtime


# -- step_fetch_data -------------------------------------------------------


def test_fetch_data_skips_when_file_exists(tmp_path: Path) -> None:
    """step_fetch_data skips download when PGN file already exists."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    # Pre-create the PGN file
    pgn_path = Path(cfg.data.pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text("fake pgn data")

    with patch("subprocess.run") as mock_run:
        step_fetch_data(cfg, state)
        mock_run.assert_not_called()

    assert state.phase == "fetched"


def test_fetch_data_calls_wget_when_missing(tmp_path: Path) -> None:
    """step_fetch_data invokes wget when PGN file does not exist."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    with patch("subprocess.run") as mock_run:
        step_fetch_data(cfg, state)
        mock_run.assert_called_once()
        args = mock_run.call_args[0][0]
        assert args[0] == "wget"
        assert cfg.data.pgn_url in args

    assert state.phase == "fetched"
    assert state.updated_at != ""


def test_fetch_data_maps_sigint_exit_to_keyboard_interrupt(tmp_path: Path) -> None:
    """SIGINT-style wget exit code is normalized to KeyboardInterrupt."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=130,
            cmd=["wget"],
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            step_fetch_data(cfg, state)


def test_run_python_module_maps_sigint_exit_to_keyboard_interrupt() -> None:
    """SIGINT-style child exit code is normalized to KeyboardInterrupt."""
    with patch(
        "subprocess.run",
        side_effect=subprocess.CalledProcessError(
            returncode=130,
            cmd=["python", "-m", "denoisr.scripts.train_phase1"],
        ),
    ):
        with pytest.raises(KeyboardInterrupt):
            _run_python_module("denoisr.scripts.train_phase1", [])


def test_run_python_module_forces_tqdm_off() -> None:
    """Pipeline child scripts should always disable tqdm via env override."""

    def _fake_run(cmd: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        assert cmd[0] != ""
        assert env["DENOISR_TQDM"] == "0"

    with patch("subprocess.run", side_effect=_fake_run) as mock_run:
        _run_python_module("denoisr.scripts.train_phase1", ["--epochs", "1"])
        mock_run.assert_called_once()


# -- step_generate_data ----------------------------------------------------


def test_generate_data_calls_generate(tmp_path: Path) -> None:
    """step_generate_data calls generate_to_file with correct params."""
    cfg = PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            pgn_path=str(tmp_path / "data" / "lichess.pgn.zst"),
            stockfish_path="",
            stockfish_depth=10,
            max_examples=123,
            workers=4,
            chunksize=512,
            chunk_examples=2048,
        ),
        model=ModelSectionConfig(
            d_s=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            num_timesteps=10,
        ),
        phase1=Phase1Config(
            epochs=5,
            lr=3e-4,
            batch_size=32,
            holdout_frac=0.05,
            warmup_epochs=1,
            weight_decay=1e-4,
        ),
        phase2=Phase2Config(
            epochs=5,
            lr=3e-4,
            batch_size=16,
            seq_len=4,
            max_trajectories=20,
        ),
        phase3=Phase3Config(
            generations=10,
            games_per_gen=5,
            reanalyse_per_gen=2,
            mcts_sims=50,
            buffer_capacity=128,
            alpha_generations=4,
            lr=1e-4,
            train_batch_size=16,
            diffusion_steps=4,
            aux_updates_per_gen=1,
            aux_batch_size=8,
            aux_seq_len=4,
            aux_lr=1e-4,
            self_play_workers=0,
            reanalyse_workers=0,
            save_every=1,
        ),
        output=OutputConfig(
            dir=str(tmp_path / "outputs"),
            run_name="test-run",
        ),
    )
    state = PipelineState()

    with patch(
        "denoisr.scripts.generate_data.generate_to_file", return_value=100
    ) as mock_gen:
        with patch("shutil.which", return_value="/usr/bin/stockfish"):
            step_generate_data(cfg, state)
            mock_gen.assert_called_once()
            kwargs = mock_gen.call_args.kwargs
            assert kwargs["max_examples"] == 123
            assert kwargs["stockfish_path"] == "/usr/bin/stockfish"
            assert kwargs["chunksize"] == 512
            assert kwargs["chunk_examples"] == 2048
            assert kwargs["use_tqdm"] is False

    assert state.last_data != ""
    assert state.updated_at != ""


def test_generate_data_skips_when_exists(tmp_path: Path) -> None:
    """step_generate_data skips when output .pt file exists."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "training_data.pt"
    data_path.write_text("fake")

    with patch("denoisr.scripts.generate_data.generate_to_file") as mock_gen:
        step_generate_data(cfg, state)
        mock_gen.assert_not_called()

    assert state.last_data == str(data_path)


def test_generate_data_fails_fast_when_stockfish_missing(tmp_path: Path) -> None:
    """Missing Stockfish should raise a clear error before worker startup."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()
    pgn_path = Path(cfg.data.pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text("fake pgn")

    with (
        patch("denoisr.scripts.generate_data.generate_to_file") as mock_gen,
        patch("shutil.which", return_value=None),
    ):
        with pytest.raises(FileNotFoundError, match="Stockfish not found in PATH"):
            step_generate_data(cfg, state)
        mock_gen.assert_not_called()


# -- step_train_phase1 -----------------------------------------------------


def test_train_phase1_updates_state(tmp_path: Path) -> None:
    """Phase 1 step launches training and updates pipeline state."""
    cfg = _make_cfg(tmp_path)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    init_ckpt = output_dir / "init_model.pt"
    init_ckpt.write_bytes(b"init")
    data_path = output_dir / "training_data.pt"
    data_path.write_bytes(b"data")
    state = PipelineState(
        phase="model_initialized",
        last_checkpoint=str(init_ckpt),
        last_data=str(data_path),
    )

    def _fake_run(cmd: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        assert "denoisr.scripts.train_phase1" in cmd
        assert "--holdout-frac" in cmd
        assert "--epochs" in cmd
        assert "--tqdm" not in cmd
        assert env["DENOISR_TQDM"] == "0"
        (output_dir / "phase1.pt").write_bytes(b"phase1")

    with patch("subprocess.run", side_effect=_fake_run) as mock_run:
        step_train_phase1(cfg, state)
        mock_run.assert_called_once()

    assert state.phase == "phase1_complete"
    assert state.last_checkpoint == str(output_dir / "phase1.pt")
    assert state.updated_at != ""


# -- step_train_phase2 -----------------------------------------------------


def test_train_phase2_updates_state(tmp_path: Path) -> None:
    """Phase 2 step launches training and updates pipeline state."""
    cfg = _make_cfg(tmp_path)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase1_ckpt = output_dir / "phase1.pt"
    phase1_ckpt.write_bytes(b"phase1")
    pgn_path = Path(cfg.data.pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text("fake pgn")
    state = PipelineState(
        phase="phase1_complete",
        last_checkpoint=str(phase1_ckpt),
    )

    def _fake_run(cmd: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        assert "denoisr.scripts.train_phase2" in cmd
        assert "--tqdm" not in cmd
        assert env["DENOISR_TQDM"] == "0"
        (output_dir / "phase2.pt").write_bytes(b"phase2")

    with patch("subprocess.run", side_effect=_fake_run) as mock_run:
        step_train_phase2(cfg, state)
        mock_run.assert_called_once()

    assert state.phase == "phase2_complete"
    assert state.last_checkpoint == str(output_dir / "phase2.pt")
    assert state.updated_at != ""


# -- step_train_phase3 -----------------------------------------------------


def test_train_phase3_updates_state(tmp_path: Path) -> None:
    """Phase 3 step launches training and updates pipeline state."""
    cfg = _make_cfg(tmp_path)
    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase2_ckpt = output_dir / "phase2.pt"
    phase2_ckpt.write_bytes(b"phase2")
    state = PipelineState(
        phase="phase2_complete",
        last_checkpoint=str(phase2_ckpt),
    )

    def _fake_run(cmd: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        assert "denoisr.scripts.train_phase3" in cmd
        assert "--tqdm" not in cmd
        assert env["DENOISR_TQDM"] == "0"
        (output_dir / "phase3.pt").write_bytes(b"phase3")

    with patch("subprocess.run", side_effect=_fake_run) as mock_run:
        step_train_phase3(cfg, state)
        mock_run.assert_called_once()

    assert state.phase == "phase3_complete"
    assert state.last_checkpoint == str(output_dir / "phase3.pt")
    assert state.updated_at != ""


# -- State round-trip after steps ------------------------------------------


def test_state_persists_after_init_model(tmp_path: Path) -> None:
    """State changes from step_init_model survive save/load."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    step_init_model(cfg, state)

    state_path = tmp_path / "state.json"
    state.save(state_path)
    loaded = PipelineState.load(state_path)

    assert loaded.phase == "model_initialized"
    assert loaded.last_checkpoint == state.last_checkpoint
    assert loaded.updated_at == state.updated_at


def test_state_persists_full_pipeline_phases(tmp_path: Path) -> None:
    """State survives save/load after cycling through all training phases."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    # Simulate the full phase progression
    step_init_model(cfg, state)
    output_dir = Path(cfg.output.dir)
    data_path = output_dir / "training_data.pt"
    data_path.write_bytes(b"data")
    state.last_data = str(data_path)
    pgn_path = Path(cfg.data.pgn_path)
    pgn_path.parent.mkdir(parents=True, exist_ok=True)
    pgn_path.write_text("fake pgn")

    def _fake_run(cmd: list[str], check: bool, env: dict[str, str]) -> None:
        assert check
        cmd_str = " ".join(cmd)
        assert "--tqdm" not in cmd
        assert env["DENOISR_TQDM"] == "0"
        if "denoisr.scripts.train_phase1" in cmd_str:
            (output_dir / "phase1.pt").write_bytes(b"phase1")
            return
        if "denoisr.scripts.train_phase2" in cmd_str:
            (output_dir / "phase2.pt").write_bytes(b"phase2")
            return
        if "denoisr.scripts.train_phase3" in cmd_str:
            (output_dir / "phase3.pt").write_bytes(b"phase3")
            return
        raise AssertionError(f"unexpected command: {cmd_str}")

    with patch("subprocess.run", side_effect=_fake_run):
        step_train_phase1(cfg, state)
        step_train_phase2(cfg, state)
        step_train_phase3(cfg, state)

    # Save and reload
    state_path = tmp_path / "state.json"
    state.save(state_path)
    loaded = PipelineState.load(state_path)

    assert loaded.phase == "phase3_complete"
    assert loaded.last_checkpoint != ""
