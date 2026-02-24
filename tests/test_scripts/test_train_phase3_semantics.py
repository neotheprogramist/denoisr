import pytest

import denoisr.scripts.train_phase3 as train_phase3


def test_phase3_parser_fails_fast_when_required_args_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
    ]
    for key in required_envs:
        monkeypatch.delenv(key, raising=False)

    parser = train_phase3.build_cli_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--checkpoint", "outputs/phase2.pt"])


def test_phase3_parser_accepts_env_backed_required_args(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
