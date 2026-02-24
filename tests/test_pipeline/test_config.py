import os

import pytest

from denoisr.pipeline.config import load_config


def _clear_denoisr_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("DENOISR_"):
            monkeypatch.delenv(key, raising=False)


def test_load_defaults_without_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_denoisr_env(monkeypatch)
    cfg = load_config()
    assert cfg.data.stockfish_depth == 10
    assert cfg.data.max_examples == 4_000_000
    assert cfg.data.workers == 64
    assert cfg.data.chunksize == 1_024
    assert cfg.data.chunk_examples == 1_000_000
    assert cfg.phase1.lr == 3e-4
    assert cfg.phase2.epochs == 120


def test_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_denoisr_env(monkeypatch)
    monkeypatch.setenv("DENOISR_PGN_PATH", "data/custom.pgn.zst")
    monkeypatch.setenv("DENOISR_STOCKFISH_DEPTH", "15")
    monkeypatch.setenv("DENOISR_MAX_EXAMPLES", "500_000")
    monkeypatch.setenv("DENOISR_WORKERS", "32")
    monkeypatch.setenv("DENOISR_CHUNKSIZE", "256")
    monkeypatch.setenv("DENOISR_CHUNK_EXAMPLES", "4096")
    monkeypatch.setenv("DENOISR_MODEL_D_S", "512")
    monkeypatch.setenv("DENOISR_PHASE1_LR", "5e-4")
    monkeypatch.setenv("DENOISR_PHASE2_EPOCHS", "300")
    cfg = load_config()
    assert cfg.data.pgn_path == "data/custom.pgn.zst"
    assert cfg.data.stockfish_depth == 15
    assert cfg.data.max_examples == 500_000
    assert cfg.data.workers == 32
    assert cfg.data.chunksize == 256
    assert cfg.data.chunk_examples == 4096
    assert cfg.model.d_s == 512
    assert cfg.phase1.lr == 5e-4
    assert cfg.phase2.epochs == 300
    assert cfg.phase1.batch_size == 128
    assert cfg.output.dir == "outputs/"


def test_invalid_env_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_denoisr_env(monkeypatch)
    monkeypatch.setenv("DENOISR_PHASE1_BATCH_SIZE", "not_an_int")
    with pytest.raises(ValueError, match="DENOISR_PHASE1_BATCH_SIZE"):
        load_config()
