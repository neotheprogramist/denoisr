"""Tests for PipelineRunner."""

from pathlib import Path
from unittest.mock import patch

from denoisr.pipeline.config import (
    DataConfig,
    ModelSectionConfig,
    OutputConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    PipelineConfig,
)
from denoisr.pipeline.runner import ALL_STEPS, PipelineRunner
from denoisr.pipeline.state import PipelineState


def _make_cfg(tmp_path: Path) -> PipelineConfig:
    """Build a PipelineConfig pointing at tmp_path for all I/O."""
    return PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            pgn_path=str(tmp_path / "data" / "lichess.pgn.zst"),
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


_STEP_PREFIX = "denoisr.pipeline.runner"


def _patch_all_steps():
    """Return a dict of patchers for all step functions in the runner module."""
    names = [
        "step_fetch_data",
        "step_init_model",
        "step_generate_data",
        "step_train_phase1",
        "step_train_phase2",
        "step_train_phase3",
    ]
    return {name: patch(f"{_STEP_PREFIX}.{name}") for name in names}


# -- Constructor / state file creation -------------------------------------


def test_runner_creates_state_file(tmp_path: Path) -> None:
    """PipelineRunner.__init__ persists an initial state file."""
    cfg = _make_cfg(tmp_path)
    runner = PipelineRunner(cfg)
    # State file does not exist until run() or _save_state() is called,
    # but run() calls _save_state() at the start, so call run with mocks.
    patchers = _patch_all_steps()
    mocks = {k: p.start() for k, p in patchers.items()}
    try:
        runner.run()
    finally:
        for p in patchers.values():
            p.stop()

    assert runner.state_path.exists()
    loaded = PipelineState.load(runner.state_path)
    assert loaded.started_at != ""
    assert loaded.updated_at != ""


def test_runner_restart_ignores_existing_state(tmp_path: Path) -> None:
    """PipelineRunner with restart=True creates fresh state even when a state file exists."""
    cfg = _make_cfg(tmp_path)

    # Create an existing state with progress
    state_path = Path(cfg.output.dir) / "pipeline_state.json"
    old_state = PipelineState(
        phase="phase2_complete",
        started_at="2024-01-01T00:00:00+00:00",
    )
    old_state.save(state_path)

    runner = PipelineRunner(cfg, restart=True)
    assert runner.state.phase == "init"


# -- Only-step filtering ---------------------------------------------------


def test_runner_only_runs_selected_steps(tmp_path: Path) -> None:
    """PipelineRunner with only={'init'} runs only the init step."""
    cfg = _make_cfg(tmp_path)
    runner = PipelineRunner(cfg, only=frozenset({"init"}))

    patchers = _patch_all_steps()
    mocks = {k: p.start() for k, p in patchers.items()}
    try:
        runner.run()
    finally:
        for p in patchers.values():
            p.stop()

    mocks["step_fetch_data"].assert_not_called()
    mocks["step_init_model"].assert_called_once()
    mocks["step_generate_data"].assert_not_called()
    mocks["step_train_phase1"].assert_not_called()
    mocks["step_train_phase2"].assert_not_called()
    mocks["step_train_phase3"].assert_not_called()


def test_runner_default_only_includes_all_steps() -> None:
    """ALL_STEPS contains the five canonical step names."""
    assert ALL_STEPS == frozenset(
        {"fetch", "init", "phase1", "phase2", "phase3"}
    )


# -- Resume from saved state -----------------------------------------------


def test_runner_resumes_from_saved_state(tmp_path: Path) -> None:
    """PipelineRunner picks up where it left off when state exists on disk."""
    cfg = _make_cfg(tmp_path)

    # Simulate state left after phase1 completed
    state_path = Path(cfg.output.dir) / "pipeline_state.json"
    saved = PipelineState(
        phase="phase1_complete",
        last_checkpoint="outputs/init_model.pt",
        started_at="2024-01-01T00:00:00+00:00",
    )
    saved.save(state_path)

    runner = PipelineRunner(cfg)
    assert runner.state.phase == "phase1_complete"

    patchers = _patch_all_steps()
    mocks = {k: p.start() for k, p in patchers.items()}

    def _fake_p2(c, s):
        s.phase = "phase2_complete"

    def _fake_p3(c, s):
        s.phase = "phase3_complete"

    mocks["step_train_phase2"].side_effect = _fake_p2
    mocks["step_train_phase3"].side_effect = _fake_p3

    try:
        runner.run()
    finally:
        for p in patchers.values():
            p.stop()

    # fetch/init/phase1 should be skipped because phase is "phase1_complete"
    mocks["step_fetch_data"].assert_not_called()
    mocks["step_init_model"].assert_not_called()
    mocks["step_generate_data"].assert_not_called()
    mocks["step_train_phase1"].assert_not_called()

    # Phase 2 and 3 should run
    mocks["step_train_phase2"].assert_called_once()
    mocks["step_train_phase3"].assert_called_once()


# -- Full pipeline run (all steps mocked) ----------------------------------


def test_runner_full_pipeline(tmp_path: Path) -> None:
    """PipelineRunner calls all steps in correct order for a fresh run."""
    cfg = _make_cfg(tmp_path)
    runner = PipelineRunner(cfg)

    patchers = _patch_all_steps()
    mocks = {k: p.start() for k, p in patchers.items()}

    # Make mocked steps advance state phases so downstream guards pass
    def _fake_fetch(c, s):
        s.phase = "fetched"

    def _fake_init(c, s):
        s.phase = "model_initialized"

    def _fake_gen(c, s):
        pass

    def _fake_p1(c, s):
        s.phase = "phase1_complete"

    def _fake_p2(c, s):
        s.phase = "phase2_complete"

    def _fake_p3(c, s):
        s.phase = "phase3_complete"

    mocks["step_fetch_data"].side_effect = _fake_fetch
    mocks["step_init_model"].side_effect = _fake_init
    mocks["step_generate_data"].side_effect = _fake_gen
    mocks["step_train_phase1"].side_effect = _fake_p1
    mocks["step_train_phase2"].side_effect = _fake_p2
    mocks["step_train_phase3"].side_effect = _fake_p3

    try:
        runner.run()
    finally:
        for p in patchers.values():
            p.stop()

    mocks["step_fetch_data"].assert_called_once()
    mocks["step_init_model"].assert_called_once()
    mocks["step_generate_data"].assert_called_once()
    mocks["step_train_phase1"].assert_called_once()
    mocks["step_train_phase2"].assert_called_once()
    mocks["step_train_phase3"].assert_called_once()
    assert runner.state.phase == "phase3_complete"


# -- Phase-based skip logic ------------------------------------------------


def test_runner_skips_phase2_when_already_complete(tmp_path: Path) -> None:
    """Phase 2 is skipped if state.phase is already 'phase2_complete'."""
    cfg = _make_cfg(tmp_path)

    state_path = Path(cfg.output.dir) / "pipeline_state.json"
    saved = PipelineState(
        phase="phase2_complete",
        started_at="2024-01-01T00:00:00+00:00",
    )
    saved.save(state_path)

    runner = PipelineRunner(cfg)

    patchers = _patch_all_steps()
    mocks = {k: p.start() for k, p in patchers.items()}

    def _fake_p3(c, s):
        s.phase = "phase3_complete"

    mocks["step_train_phase3"].side_effect = _fake_p3

    try:
        runner.run()
    finally:
        for p in patchers.values():
            p.stop()

    mocks["step_train_phase2"].assert_not_called()
    mocks["step_train_phase3"].assert_called_once()
