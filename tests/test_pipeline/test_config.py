from pathlib import Path

import pytest

from denoisr.pipeline.config import load_config


def test_load_minimal_config(tmp_path: Path) -> None:
    """Empty TOML uses all defaults."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("")
    cfg = load_config(cfg_path)
    assert cfg.data.stockfish_depth == 10
    assert cfg.data.chunk_examples == 250_000
    assert cfg.phase1.lr == 3e-4
    assert cfg.phase2.epochs == 120


def test_load_partial_config(tmp_path: Path) -> None:
    """Partial TOML overrides only specified fields."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[phase1]\nlr = 1e-3\n")
    cfg = load_config(cfg_path)
    assert cfg.phase1.lr == 1e-3
    assert cfg.phase1.batch_size == 128  # default preserved


def test_full_config(tmp_path: Path) -> None:
    """All sections specified."""
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("""
[data]
stockfish_depth = 15
max_examples = 500_000
chunk_examples = 4096

[model]
d_s = 512

[phase1]
lr = 5e-4

[phase2]
epochs = 300
""")
    cfg = load_config(cfg_path)
    assert cfg.data.stockfish_depth == 15
    assert cfg.data.max_examples == 500_000
    assert cfg.data.chunk_examples == 4096
    assert cfg.model.d_s == 512
    assert cfg.phase1.lr == 5e-4
    assert cfg.phase2.epochs == 300
    # Defaults preserved
    assert cfg.phase1.batch_size == 128
    assert cfg.output.dir == "outputs/"


def test_load_without_toml_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DENOISR_PGN_PATH", "data/custom.pgn.zst")
    monkeypatch.setenv("DENOISR_PHASE1_BATCH_SIZE", "96")
    monkeypatch.setenv("DENOISR_PHASE2_EPOCHS", "90")
    monkeypatch.setenv("DENOISR_PHASE3_MCTS_SIMS", "256")
    monkeypatch.setenv("DENOISR_OUTPUT_DIR", "runs/")

    cfg = load_config(None)
    assert cfg.data.pgn_path == "data/custom.pgn.zst"
    assert cfg.phase1.batch_size == 96
    assert cfg.phase2.epochs == 90
    assert cfg.phase3.mcts_sims == 256
    assert cfg.output.dir == "runs/"
