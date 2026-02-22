"""Integration tests for the pipeline runner.

These tests exercise real step logic where feasible (e.g. model init) and
mock only external dependencies (wget, sort_pgn, generate_to_file) that
require Stockfish or large data files.
"""

from pathlib import Path
from unittest.mock import patch

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
from denoisr.pipeline.runner import PipelineRunner
from denoisr.pipeline.state import PipelineState


def _make_cfg(tmp_path: Path) -> PipelineConfig:
    """Build a PipelineConfig with small dimensions pointing at tmp_path."""
    return PipelineConfig(
        data=DataConfig(
            pgn_url="https://example.com/test.pgn.zst",
            data_dir=str(tmp_path / "data"),
            stockfish_path="",
            stockfish_depth=1,
            examples_per_tier=10,
            workers=1,
        ),
        elo_curriculum=EloCurriculumConfig(
            tiers=[800, 1200],
            gate_accuracy=0.50,
            max_epochs_per_tier=2,
        ),
        model=ModelSectionConfig(
            d_s=64,
            num_heads=4,
            num_layers=2,
            ffn_dim=128,
            num_timesteps=10,
        ),
        phase1=Phase1Config(lr=3e-4, batch_size=32),
        phase2=Phase2Config(epochs=2, lr=3e-4, batch_size=16, seq_len=4),
        phase3=Phase3Config(generations=2, games_per_gen=2, mcts_sims=10),
        output=OutputConfig(dir=str(tmp_path / "outputs")),
    )


_RUNNER = "denoisr.pipeline.runner"


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
        # in the set {"sorted", "fetched", "init", ""} checked by the runner
        mock_init.assert_not_called()


# ---------------------------------------------------------------------------
# 2. Full placeholder run
# ---------------------------------------------------------------------------


def test_pipeline_full_placeholder_run(tmp_path: Path) -> None:
    """Run the full pipeline with external steps mocked; verify phase3_complete."""
    cfg = _make_cfg(tmp_path)

    # Mock only external-facing steps: fetch, sort, generate_data.
    # step_init_model is real (creates a tiny model), and the phase1/2/3
    # training placeholders are real (they just update state).
    def _fake_fetch(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "fetched"

    def _fake_sort(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "sorted"

    def _fake_generate(
        c: PipelineConfig, s: PipelineState, min_elo: int, tier_index: int
    ) -> None:
        out = Path(c.output.dir) / f"tier_{tier_index}_elo{min_elo}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        s.last_data = str(out)

    with (
        patch(f"{_RUNNER}.step_fetch_data", side_effect=_fake_fetch),
        patch(f"{_RUNNER}.step_sort_pgn", side_effect=_fake_sort),
        patch(
            f"{_RUNNER}.step_generate_tier_data", side_effect=_fake_generate
        ),
    ):
        runner = PipelineRunner(cfg)
        runner.run()

    # Final state should be phase3_complete
    assert runner.state.phase == "phase3_complete"

    # All tiers should have accuracy recorded
    tiers = cfg.elo_curriculum.tiers
    for elo in tiers:
        assert str(elo) in runner.state.tier_accuracies

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
        elo_tier_index=2,
        last_checkpoint="outputs/some_checkpoint.pt",
        tier_accuracies={"800": 0.55, "1200": 0.52},
        started_at="2024-01-01T00:00:00+00:00",
        updated_at="2024-06-15T12:00:00+00:00",
    )
    old_state.save(state_path)
    assert state_path.exists()

    # Create runner with restart=True
    runner = PipelineRunner(cfg, restart=True)

    # State should be completely fresh
    assert runner.state.phase == "init"
    assert runner.state.elo_tier_index == 0
    assert runner.state.last_checkpoint == ""
    assert runner.state.tier_accuracies == {}

    # started_at should be set to a new timestamp (not the old one)
    assert runner.state.started_at != ""
    assert runner.state.started_at != "2024-01-01T00:00:00+00:00"

    # Running with all mocks should start from scratch
    def _fake_fetch(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "fetched"

    def _fake_sort(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "sorted"

    def _fake_init(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "model_initialized"

    def _fake_generate(
        c: PipelineConfig, s: PipelineState, min_elo: int, tier_index: int
    ) -> None:
        pass

    def _fake_p1(
        c: PipelineConfig, s: PipelineState, tier_index: int, min_elo: int
    ) -> None:
        s.elo_tier_index = tier_index + 1
        s.phase = "elo_curriculum"

    def _fake_p2(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "phase2_complete"

    def _fake_p3(c: PipelineConfig, s: PipelineState) -> None:
        s.phase = "phase3_complete"

    with (
        patch(f"{_RUNNER}.step_fetch_data", side_effect=_fake_fetch) as m_fetch,
        patch(f"{_RUNNER}.step_sort_pgn", side_effect=_fake_sort) as m_sort,
        patch(f"{_RUNNER}.step_init_model", side_effect=_fake_init) as m_init,
        patch(
            f"{_RUNNER}.step_generate_tier_data", side_effect=_fake_generate
        ) as m_gen,
        patch(
            f"{_RUNNER}.step_train_phase1_tier", side_effect=_fake_p1
        ) as m_p1,
        patch(f"{_RUNNER}.step_train_phase2", side_effect=_fake_p2) as m_p2,
        patch(f"{_RUNNER}.step_train_phase3", side_effect=_fake_p3) as m_p3,
    ):
        runner.run()

    # All steps should have been called (fresh start)
    m_fetch.assert_called_once()
    m_sort.assert_called_once()
    m_init.assert_called_once()
    assert m_gen.call_count == 2  # 2 tiers
    assert m_p1.call_count == 2
    m_p2.assert_called_once()
    m_p3.assert_called_once()

    assert runner.state.phase == "phase3_complete"


# ---------------------------------------------------------------------------
# 4. Only phase1
# ---------------------------------------------------------------------------


def test_pipeline_only_phase1(tmp_path: Path) -> None:
    """Run with only=frozenset({'phase1'}), verify only phase1 tier steps called."""
    cfg = _make_cfg(tmp_path)

    # Pre-seed state so phase1 guards pass (the runner phase1 block does not
    # check phase for entry, it runs unconditionally when in the only set).
    # We still need mocks for generate_tier_data (external dependency).
    def _fake_generate(
        c: PipelineConfig, s: PipelineState, min_elo: int, tier_index: int
    ) -> None:
        out = Path(c.output.dir) / f"tier_{tier_index}_elo{min_elo}.pt"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(b"fake")
        s.last_data = str(out)

    with (
        patch(f"{_RUNNER}.step_fetch_data") as m_fetch,
        patch(f"{_RUNNER}.step_sort_pgn") as m_sort,
        patch(f"{_RUNNER}.step_init_model") as m_init,
        patch(
            f"{_RUNNER}.step_generate_tier_data", side_effect=_fake_generate
        ) as m_gen,
        patch(f"{_RUNNER}.step_train_phase2") as m_p2,
        patch(f"{_RUNNER}.step_train_phase3") as m_p3,
    ):
        runner = PipelineRunner(cfg, only=frozenset({"phase1"}))
        runner.run()

    # Fetch, sort, init should NOT be called
    m_fetch.assert_not_called()
    m_sort.assert_not_called()
    m_init.assert_not_called()

    # Phase 2 and phase 3 should NOT be called
    m_p2.assert_not_called()
    m_p3.assert_not_called()

    # Generate + phase1 training should be called for each tier
    assert m_gen.call_count == 2
    assert runner.state.elo_tier_index == 2
    assert runner.state.phase == "elo_curriculum"

    # Verify tier accuracies were recorded by the real step_train_phase1_tier
    for elo in cfg.elo_curriculum.tiers:
        assert str(elo) in runner.state.tier_accuracies
