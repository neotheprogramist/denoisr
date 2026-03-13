"""Integration tests for the pipeline runner.

These tests exercise real step logic where feasible (e.g. model init) and
mock only external dependencies (wget, generate_to_file) that require
Stockfish or large data files.
"""

from pathlib import Path
from unittest.mock import patch

from denoisr_chess.pipeline.config import (
    DataConfig,
    ModelSectionConfig,
    OutputConfig,
    Phase1Config,
    Phase2Config,
    Phase3Config,
    PipelineConfig,
)
from denoisr_chess.pipeline.runner import PipelineRunner
from denoisr_chess.pipeline.state import PipelineState


def _make_cfg(tmp_path: Path) -> PipelineConfig:
    """Build a PipelineConfig with small dimensions pointing at tmp_path."""
    return PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            pgn_path=str(tmp_path / "data" / "lichess.pgn.zst"),
            stockfish_path="stockfish",
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
            epochs=2,
            lr=3e-4,
            batch_size=32,
            holdout_frac=0.05,
            warmup_epochs=1,
            weight_decay=1e-4,
        ),
        phase2=Phase2Config(
            epochs=2,
            lr=3e-4,
            batch_size=16,
            seq_len=4,
            max_trajectories=20,
        ),
        phase3=Phase3Config(
            generations=2,
            games_per_gen=2,
            reanalyse_per_gen=1,
            mcts_sims=10,
            buffer_capacity=64,
            alpha_generations=2,
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


_RUNNER = "denoisr_chess.pipeline.runner"


# ---------------------------------------------------------------------------
# 1. Init model and resume
# ---------------------------------------------------------------------------


def test_pipeline_init_model_and_resume(tmp_path: Path) -> None:
    """Run with only='init', verify checkpoint, then resume and verify init skipped."""
    cfg = _make_cfg(tmp_path)

    # --- First run: only init ---
    runner = PipelineRunner(cfg, only=frozenset({"init"}))
    runner.run()

    # Verify a checkpoint file was created on disk
    ckpt_path = Path(runner.state.last_checkpoint)
    assert ckpt_path.exists(), f"Checkpoint not found at {ckpt_path}"
    assert runner.state.phase == "model_initialized"

    # State file was persisted
    assert runner.state_path.exists()

    # --- Second run: resume (re-create runner, no restart) ---
    runner2 = PipelineRunner(cfg, only=frozenset({"init"}))
    assert runner2.state.phase == "model_initialized", (
        "Resumed runner should load persisted phase"
    )
    assert runner2.state.last_checkpoint == str(ckpt_path)

    # Patch step_init_model to detect whether it is called
    with patch(f"{_RUNNER}.step_init_model", wraps=None) as mock_init:
        runner2.run()
        # init should be skipped: phase is "model_initialized" which is not
        # in the set {"fetched", "init", ""} checked by the runner
        mock_init.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Full pipeline run with training steps mocked
# ---------------------------------------------------------------------------


def test_pipeline_full_run_with_mocked_training(tmp_path: Path) -> None:
    """Run the full pipeline with data/training externalities mocked."""
    cfg = _make_cfg(tmp_path)

    # Mock external-facing steps.
    # step_init_model is real (creates a tiny model).
    def _fake_fetch(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "fetched"

    def _fake_generate(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "training_data.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        s.last_data = str(out)

    def _fake_p1(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "phase1.pt"
        out.write_bytes(b"phase1")
        s.last_checkpoint = str(out)
        s.phase = "phase1_complete"

    def _fake_p2(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "phase2.pt"
        out.write_bytes(b"phase2")
        s.last_checkpoint = str(out)
        s.phase = "phase2_complete"

    def _fake_p3(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "phase3.pt"
        out.write_bytes(b"phase3")
        s.last_checkpoint = str(out)
        s.phase = "phase3_complete"

    with (
        patch(f"{_RUNNER}.step_fetch_data", side_effect=_fake_fetch),
        patch(f"{_RUNNER}.step_generate_data", side_effect=_fake_generate),
        patch(f"{_RUNNER}.step_train_phase1", side_effect=_fake_p1),
        patch(f"{_RUNNER}.step_train_phase2", side_effect=_fake_p2),
        patch(f"{_RUNNER}.step_train_phase3", side_effect=_fake_p3),
    ):
        runner = PipelineRunner(cfg)
        runner.run()

    # Final state should be phase3_complete
    assert runner.state.phase == "phase3_complete"

    # Checkpoint should exist (from real step_init_model)
    assert runner.state.last_checkpoint != ""
    assert Path(runner.state.last_checkpoint).exists()

    # State file should be persisted
    loaded = PipelineState.load(runner.state_path)
    assert loaded.phase == "phase3_complete"


# ---------------------------------------------------------------------------
# 3. Restart clears state
# ---------------------------------------------------------------------------


def test_pipeline_restart_clears_state(tmp_path: Path) -> None:
    """Create a state file with progress, run with restart=True, verify fresh state."""
    cfg = _make_cfg(tmp_path)

    # Write a state file showing progress deep into phase2
    state_path = Path(cfg.output.dir) / "pipeline_state.json"
    old_state = PipelineState(
        phase="phase2_complete",
        last_checkpoint="outputs/some_checkpoint.pt",
        started_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-06-15T12:00:00+00:00",
    )
    old_state.save(state_path)
    assert state_path.exists()

    # Create runner with restart=True
    runner = PipelineRunner(cfg, restart=True)

    # State should be completely fresh
    assert runner.state.phase == "init"
    assert runner.state.last_checkpoint == ""

    # started_at should be set to a new timestamp (not the old one)
    assert runner.state.started_at != ""
    assert runner.state.started_at != "2024-01-01T00:00:00+00:00"

    # Running with all mocks should start from scratch
    def _fake_fetch(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "fetched"

    def _fake_init(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "model_initialized"

    def _fake_generate(c: PipelineConfig, s: PipelineState) -> None:
        pass

    def _fake_p1(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "phase1_complete"

    def _fake_p2(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "phase2_complete"

    def _fake_p3(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "phase3_complete"

    with (
        patch(f"{_RUNNER}.step_fetch_data", side_effect=_fake_fetch) as m_fetch,
        patch(f"{_RUNNER}.step_init_model", side_effect=_fake_init) as m_init,
        patch(f"{_RUNNER}.step_generate_data", side_effect=_fake_generate) as m_gen,
        patch(f"{_RUNNER}.step_train_phase1", side_effect=_fake_p1) as m_p1,
        patch(f"{_RUNNER}.step_train_phase2", side_effect=_fake_p2) as m_p2,
        patch(f"{_RUNNER}.step_train_phase3", side_effect=_fake_p3) as m_p3,
    ):
        runner.run()

    # All steps should have been called (fresh start)
    m_fetch.assert_called_once()
    m_init.assert_called_once()
    m_gen.assert_called_once()
    m_p1.assert_called_once()
    m_p2.assert_called_once()
    m_p3.assert_called_once()

    assert runner.state.phase == "phase3_complete"


# ---------------------------------------------------------------------------
# 4. Only phase1
# ---------------------------------------------------------------------------


def test_pipeline_only_phase1(tmp_path: Path) -> None:
    """Run with only=frozenset({'phase1'}), verify only phase1 steps called."""
    cfg = _make_cfg(tmp_path)

    def _fake_generate(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "training_data.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        s.last_data = str(out)

    def _fake_p1(c: PipelineConfig, s: PipelineState) -> None:
        out = Path(c.output.dir) / "phase1.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"phase1")
        s.last_checkpoint = str(out)
        s.phase = "phase1_complete"

    with (
        patch(f"{_RUNNER}.step_fetch_data") as m_fetch,
        patch(f"{_RUNNER}.step_init_model") as m_init,
        patch(f"{_RUNNER}.step_generate_data", side_effect=_fake_generate) as m_gen,
        patch(f"{_RUNNER}.step_train_phase1", side_effect=_fake_p1) as m_p1,
        patch(f"{_RUNNER}.step_train_phase2") as m_p2,
        patch(f"{_RUNNER}.step_train_phase3") as m_p3,
    ):
        runner = PipelineRunner(cfg, only=frozenset({"phase1"}))
        runner.run()

    # Fetch and init should NOT be called
    m_fetch.assert_not_called()
    m_init.assert_not_called()

    # Phase 2 and phase 3 should NOT be called
    m_p2.assert_not_called()
    m_p3.assert_not_called()

    # Generate + phase1 training should be called
    m_gen.assert_called_once()
    m_p1.assert_called_once()
    assert runner.state.phase == "phase1_complete"
