"""Tests for pipeline step functions."""

from pathlib import Path
from unittest.mock import patch

import torch

from denoisr.pipeline.config import (
    DataConfig,
    EloCurriculumConfig,
    ModelSectionConfig,
    OutputConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    PipelineConfig,
)
from denoisr.pipeline.state import PipelineState
from denoisr.pipeline.steps import (
    step_fetch_data,
    step_generate_tier_data,
    step_init_model,
    step_sort_pgn,
    step_train_phase1_tier,
    step_train_phase2,
    step_train_phase3,
)


def _make_cfg(tmp_path: Path) -> PipelineConfig:
    """Build a PipelineConfig pointing at tmp_path for all I/O."""
    return PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            data_dir=str(tmp_path / "data"),
        ),
        elo_curriculum=EloCurriculumConfig(
            tiers=[800, 1200, 1600],
            gate_accuracy=0.50,
            max_epochs_per_tier=10,
        ),
        model=ModelSectionConfig(
            d_s=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            num_timesteps=10,
        ),
        phase1=Phase1Config(lr=3e-4, batch_size=32),
        phase2=Phase2Config(epochs=5, lr=3e-4, batch_size=16, seq_len=4),
        phase3=Phase3Config(generations=10, games_per_gen=5, mcts_sims=50),
        output=OutputConfig(dir=str(tmp_path / "outputs")),
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

    # Pre-create the PGN file at data_dir / "raw.pgn.zst"
    pgn_path = Path(cfg.data.data_dir) / "raw.pgn.zst"
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


# -- step_sort_pgn ---------------------------------------------------------


def test_sort_pgn_skips_when_games_files_exist(tmp_path: Path) -> None:
    """step_sort_pgn skips when the data directory already has .games files."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    # Pre-create data dir with a fake .games bucket file
    data_dir = Path(cfg.data.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "0-800.games").write_bytes(b"fake")

    with patch("denoisr.scripts.sort_pgn.sort_pgn_to_games") as mock_sort:
        step_sort_pgn(cfg, state)
        mock_sort.assert_not_called()

    assert state.phase == "sorted"


def test_sort_pgn_calls_sort_when_no_games_files(tmp_path: Path) -> None:
    """step_sort_pgn calls sort_pgn_to_games() when no .games files exist."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    with patch("denoisr.scripts.sort_pgn.sort_pgn_to_games") as mock_sort:
        step_sort_pgn(cfg, state)
        mock_sort.assert_called_once()

    assert state.phase == "sorted"
    assert state.updated_at != ""


# -- step_generate_tier_data -----------------------------------------------


def test_generate_tier_data_calls_generate(tmp_path: Path) -> None:
    """step_generate_tier_data calls generate_to_file with correct params."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    with patch("denoisr.scripts.generate_data.generate_to_file", return_value=100) as mock_gen:
        step_generate_tier_data(cfg, state, min_elo=800, tier_index=0)
        mock_gen.assert_called_once()
        call_kwargs = mock_gen.call_args
        assert call_kwargs.kwargs["min_elo"] == 800
        assert call_kwargs.kwargs["data_dir"] == Path(cfg.data.data_dir)

    assert state.last_data != ""
    assert state.updated_at != ""


def test_generate_tier_data_skips_when_exists(tmp_path: Path) -> None:
    """step_generate_tier_data skips when output .pt file exists."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    output_dir = Path(cfg.output.dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_path = output_dir / "tier_0_elo800.pt"
    data_path.write_text("fake")

    with patch("denoisr.scripts.generate_data.generate_to_file") as mock_gen:
        step_generate_tier_data(cfg, state, min_elo=800, tier_index=0)
        mock_gen.assert_not_called()

    assert state.last_data == str(data_path)


# -- step_train_phase1_tier ------------------------------------------------


def test_train_phase1_tier_updates_state(tmp_path: Path) -> None:
    """Placeholder phase1 training updates tier accuracy and advances index."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState(phase="model_initialized", elo_tier_index=0)

    step_train_phase1_tier(cfg, state, tier_index=0, min_elo=800)

    assert state.elo_tier_index == 1
    assert "800" in state.tier_accuracies
    assert state.tier_accuracies["800"] == cfg.elo_curriculum.gate_accuracy
    assert state.phase == "elo_curriculum"
    assert state.updated_at != ""


def test_train_phase1_tier_multiple_tiers(tmp_path: Path) -> None:
    """Running phase1 across multiple tiers accumulates tier_accuracies."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState(phase="model_initialized")

    for i, elo in enumerate(cfg.elo_curriculum.tiers):
        step_train_phase1_tier(cfg, state, tier_index=i, min_elo=elo)

    assert state.elo_tier_index == len(cfg.elo_curriculum.tiers)
    assert len(state.tier_accuracies) == len(cfg.elo_curriculum.tiers)
    for elo in cfg.elo_curriculum.tiers:
        assert str(elo) in state.tier_accuracies


# -- step_train_phase2 -----------------------------------------------------


def test_train_phase2_updates_state(tmp_path: Path) -> None:
    """Placeholder phase2 training updates state phase."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState(phase="elo_curriculum")

    step_train_phase2(cfg, state)

    assert state.phase == "phase2_complete"
    assert state.updated_at != ""


# -- step_train_phase3 -----------------------------------------------------


def test_train_phase3_updates_state(tmp_path: Path) -> None:
    """Placeholder phase3 training updates state phase."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState(phase="phase2_complete")

    step_train_phase3(cfg, state)

    assert state.phase == "phase3_complete"
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
    """State survives save/load after cycling through all placeholder phases."""
    cfg = _make_cfg(tmp_path)
    state = PipelineState()

    # Simulate the full phase progression
    step_init_model(cfg, state)

    for i, elo in enumerate(cfg.elo_curriculum.tiers):
        step_train_phase1_tier(cfg, state, tier_index=i, min_elo=elo)

    step_train_phase2(cfg, state)
    step_train_phase3(cfg, state)

    # Save and reload
    state_path = tmp_path / "state.json"
    state.save(state_path)
    loaded = PipelineState.load(state_path)

    assert loaded.phase == "phase3_complete"
    assert loaded.elo_tier_index == len(cfg.elo_curriculum.tiers)
    assert len(loaded.tier_accuracies) == len(cfg.elo_curriculum.tiers)
    assert loaded.last_checkpoint != ""
