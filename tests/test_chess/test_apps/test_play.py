"""Tests for denoisr-chess-play CLI defaults."""

from __future__ import annotations

from pathlib import Path
import sys

from denoisr_chess.apps import play


def test_main_allows_missing_denoising_steps_single_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    called = False

    monkeypatch.setattr(play, "load_env_file", lambda: tmp_path / ".env")
    monkeypatch.setattr(play, "configure_logging", lambda: tmp_path / "denoisr.log")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "denoisr-chess-play",
            "--checkpoint",
            "outputs/phase1.pt",
            "--mode",
            "single",
        ],
    )

    def _fake_run_uci_loop(*, engine_select_move_fn, on_isready) -> None:
        nonlocal called
        called = True
        assert callable(engine_select_move_fn)
        assert callable(on_isready)

    monkeypatch.setattr(play, "run_uci_loop", _fake_run_uci_loop)

    play.main()

    assert called
