import os

import pytest

from denoisr_chess.pipeline.config import load_config


def _clear_denoisr_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in list(os.environ):
        if key.startswith("DENOISR_"):
            monkeypatch.delenv(key, raising=False)


def _set_required_pipeline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DENOISR_PGN_URL": "https://example.com/test.pgn.zst",
        "DENOISR_PGN_PATH": "data/test.pgn.zst",
        "DENOISR_STOCKFISH_PATH": "stockfish",
        "DENOISR_STOCKFISH_DEPTH": "10",
        "DENOISR_MAX_EXAMPLES": "4000000",
        "DENOISR_WORKERS": "64",
        "DENOISR_CHUNKSIZE": "1024",
        "DENOISR_CHUNK_EXAMPLES": "1000000",
        "DENOISR_MODEL_D_S": "256",
        "DENOISR_MODEL_NUM_HEADS": "8",
        "DENOISR_MODEL_NUM_LAYERS": "15",
        "DENOISR_MODEL_FFN_DIM": "1024",
        "DENOISR_MODEL_NUM_TIMESTEPS": "100",
        "DENOISR_PHASE1_EPOCHS": "100",
        "DENOISR_PHASE1_LR": "0.0003",
        "DENOISR_PHASE1_BATCH_SIZE": "1024",
        "DENOISR_PHASE1_HOLDOUT_FRAC": "0.05",
        "DENOISR_PHASE1_WARMUP_EPOCHS": "10",
        "DENOISR_PHASE1_WEIGHT_DECAY": "0.0001",
        "DENOISR_PHASE2_EPOCHS": "100",
        "DENOISR_PHASE2_LR": "0.0003",
        "DENOISR_PHASE2_BATCH_SIZE": "1024",
        "DENOISR_PHASE2_SEQ_LEN": "10",
        "DENOISR_PHASE2_MAX_TRAJECTORIES": "30000",
        "DENOISR_PHASE3_GENERATIONS": "400",
        "DENOISR_PHASE3_GAMES_PER_GEN": "64",
        "DENOISR_PHASE3_REANALYSE_PER_GEN": "32",
        "DENOISR_PHASE3_MCTS_SIMS": "400",
        "DENOISR_PHASE3_BUFFER_CAPACITY": "50000",
        "DENOISR_PHASE3_ALPHA_GENERATIONS": "40",
        "DENOISR_PHASE3_LR": "0.0001",
        "DENOISR_PHASE3_TRAIN_BATCH_SIZE": "128",
        "DENOISR_PHASE3_DIFFUSION_STEPS": "8",
        "DENOISR_PHASE3_AUX_UPDATES_PER_GEN": "1",
        "DENOISR_PHASE3_AUX_BATCH_SIZE": "64",
        "DENOISR_PHASE3_AUX_SEQ_LEN": "10",
        "DENOISR_PHASE3_AUX_LR": "0.0001",
        "DENOISR_PHASE3_SELF_PLAY_WORKERS": "8",
        "DENOISR_PHASE3_REANALYSE_WORKERS": "8",
        "DENOISR_PHASE3_SAVE_EVERY": "10",
        "DENOISR_OUTPUT_DIR": "outputs/",
        "DENOISR_RUN_NAME": "baseline-run",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_load_config_fails_fast_when_env_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_denoisr_env(monkeypatch)
    with pytest.raises(ValueError, match="Environment validation failed"):
        load_config()


def test_load_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_denoisr_env(monkeypatch)
    _set_required_pipeline_env(monkeypatch)
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
    assert cfg.phase1.batch_size == 1024
    assert cfg.phase2.batch_size == 1024
    assert cfg.output.dir == "outputs/"


def test_invalid_env_override_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    _clear_denoisr_env(monkeypatch)
    _set_required_pipeline_env(monkeypatch)
    monkeypatch.setenv("DENOISR_PHASE1_BATCH_SIZE", "not_an_int")
    with pytest.raises(ValueError, match="DENOISR_PHASE1_BATCH_SIZE"):
        load_config()
