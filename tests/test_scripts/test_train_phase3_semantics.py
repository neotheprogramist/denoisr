import pytest

import denoisr.scripts.train_phase3 as train_phase3


def _set_required_training_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DENOISR_TRAIN_MAX_GRAD_NORM": "5.0",
        "DENOISR_TRAIN_WEIGHT_DECAY": "0.0001",
        "DENOISR_TRAIN_ENCODER_LR_MULTIPLIER": "1.0",
        "DENOISR_TRAIN_MIN_LR": "0.000001",
        "DENOISR_TRAIN_WARMUP_EPOCHS": "10",
        "DENOISR_TRAIN_WARM_RESTARTS": "1",
        "DENOISR_TRAIN_THREAT_WEIGHT": "0.1",
        "DENOISR_TRAIN_POLICY_WEIGHT": "2.0",
        "DENOISR_TRAIN_VALUE_WEIGHT": "0.5",
        "DENOISR_TRAIN_CONSISTENCY_WEIGHT": "1.0",
        "DENOISR_TRAIN_DIFFUSION_WEIGHT": "1.0",
        "DENOISR_TRAIN_REWARD_WEIGHT": "1.0",
        "DENOISR_TRAIN_PLY_WEIGHT": "0.1",
        "DENOISR_TRAIN_ILLEGAL_PENALTY_WEIGHT": "0.01",
        "DENOISR_TRAIN_HARMONY_DREAM": "1",
        "DENOISR_TRAIN_HARMONY_EMA_DECAY": "0.99",
        "DENOISR_TRAIN_CURRICULUM_INITIAL_FRACTION": "0.25",
        "DENOISR_TRAIN_CURRICULUM_GROWTH": "1.02",
        "DENOISR_WORKERS": "64",
        "DENOISR_TQDM": "0",
        "DENOISR_PHASE1_GATE": "0.50",
        "DENOISR_PHASE2_GATE": "5.0",
        "DENOISR_GROK_TRACKING": "1",
        "DENOISR_GROK_ERANK_FREQ": "1000",
        "DENOISR_GROK_SPECTRAL_FREQ": "5000",
        "DENOISR_GROK_ONSET_THRESHOLD": "0.95",
        "DENOISR_GROKFAST": "1",
        "DENOISR_GROKFAST_ALPHA": "0.98",
        "DENOISR_GROKFAST_LAMB": "2.0",
        "DENOISR_EMA_DECAY": "0.999",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def _set_required_phase3_policy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    values = {
        "DENOISR_PHASE3_C_PUCT": "1.4",
        "DENOISR_PHASE3_DIRICHLET_ALPHA": "0.3",
        "DENOISR_PHASE3_DIRICHLET_EPSILON": "0.25",
        "DENOISR_PHASE3_TEMPERATURE_BASE": "1.0",
        "DENOISR_PHASE3_TEMPERATURE_EXPLORE_MOVES": "30",
        "DENOISR_PHASE3_TEMPERATURE_GENERATION_DECAY": "0.97",
        "DENOISR_PHASE3_MAX_MOVES": "300",
        "DENOISR_PHASE3_REANALYSE_SIMULATIONS": "100",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)


def test_phase3_parser_fails_fast_when_required_args_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_training_env(monkeypatch)
    required_envs = [
        "DENOISR_PHASE3_CHECKPOINT",
        "DENOISR_PHASE3_GENERATIONS",
        "DENOISR_PHASE3_GAMES_PER_GEN",
        "DENOISR_PHASE3_REANALYSE_PER_GEN",
        "DENOISR_PHASE3_MCTS_SIMS",
        "DENOISR_PHASE3_BUFFER_CAPACITY",
        "DENOISR_PHASE3_ALPHA_GENERATIONS",
        "DENOISR_PHASE3_LR",
        "DENOISR_PHASE3_TRAIN_BATCH_SIZE",
        "DENOISR_PHASE3_DIFFUSION_STEPS",
        "DENOISR_PHASE3_AUX_UPDATES_PER_GEN",
        "DENOISR_PHASE3_AUX_BATCH_SIZE",
        "DENOISR_PHASE3_AUX_SEQ_LEN",
        "DENOISR_PHASE3_SELF_PLAY_WORKERS",
        "DENOISR_PHASE3_REANALYSE_WORKERS",
        "DENOISR_PHASE3_OUTPUT",
        "DENOISR_PHASE3_SAVE_EVERY",
        "DENOISR_PHASE3_C_PUCT",
        "DENOISR_PHASE3_DIRICHLET_ALPHA",
        "DENOISR_PHASE3_DIRICHLET_EPSILON",
        "DENOISR_PHASE3_TEMPERATURE_BASE",
        "DENOISR_PHASE3_TEMPERATURE_EXPLORE_MOVES",
        "DENOISR_PHASE3_TEMPERATURE_GENERATION_DECAY",
        "DENOISR_PHASE3_MAX_MOVES",
        "DENOISR_PHASE3_REANALYSE_SIMULATIONS",
    ]
    for key in required_envs:
        monkeypatch.delenv(key, raising=False)

    parser = train_phase3.build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "outputs/phase2.pt"])


def test_phase3_parser_accepts_env_backed_required_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required_training_env(monkeypatch)
    _set_required_phase3_policy_env(monkeypatch)
    monkeypatch.setenv("DENOISR_PHASE3_CHECKPOINT", "outputs/phase2.pt")
    monkeypatch.setenv("DENOISR_PHASE3_GENERATIONS", "2")
    monkeypatch.setenv("DENOISR_PHASE3_GAMES_PER_GEN", "1")
    monkeypatch.setenv("DENOISR_PHASE3_REANALYSE_PER_GEN", "1")
    monkeypatch.setenv("DENOISR_PHASE3_MCTS_SIMS", "32")
    monkeypatch.setenv("DENOISR_PHASE3_BUFFER_CAPACITY", "16")
    monkeypatch.setenv("DENOISR_PHASE3_ALPHA_GENERATIONS", "2")
    monkeypatch.setenv("DENOISR_PHASE3_LR", "0.0001")
    monkeypatch.setenv("DENOISR_PHASE3_TRAIN_BATCH_SIZE", "2")
    monkeypatch.setenv("DENOISR_PHASE3_DIFFUSION_STEPS", "4")
    monkeypatch.setenv("DENOISR_PHASE3_AUX_UPDATES_PER_GEN", "1")
    monkeypatch.setenv("DENOISR_PHASE3_AUX_BATCH_SIZE", "2")
    monkeypatch.setenv("DENOISR_PHASE3_AUX_SEQ_LEN", "2")
    monkeypatch.setenv("DENOISR_PHASE3_SELF_PLAY_WORKERS", "0")
    monkeypatch.setenv("DENOISR_PHASE3_REANALYSE_WORKERS", "0")
    monkeypatch.setenv("DENOISR_PHASE3_OUTPUT", "outputs/phase3.pt")
    monkeypatch.setenv("DENOISR_PHASE3_SAVE_EVERY", "1")

    parser = train_phase3.build_cli_parser()
    args = parser.parse_args([])

    assert args.checkpoint == "outputs/phase2.pt"
    assert args.generations == 2
    assert args.games_per_gen == 1
    assert args.output == "outputs/phase3.pt"
