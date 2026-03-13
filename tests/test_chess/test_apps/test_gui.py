"""Tests for denoisr-chess-gui CLI defaults."""

from __future__ import annotations

from pathlib import Path
import sys
import types

from denoisr_chess.apps import gui


class _Var:
    def __init__(self) -> None:
        self.value = ""

    def set(self, value: str) -> None:
        self.value = value


class _DummyApp:
    last_instance: _DummyApp | None = None

    def __init__(self) -> None:
        self._engine_mode_var = _Var()
        self._ckpt_var = _Var()
        self.auto_started = False
        self.ran = False
        _DummyApp.last_instance = self

    def auto_start(self) -> None:
        self.auto_started = True

    def run(self) -> None:
        self.ran = True


def test_main_defaults_mode_to_single(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(gui, "load_env_file", lambda: tmp_path / ".env")
    monkeypatch.setattr(gui, "configure_logging", lambda: tmp_path / "denoisr.log")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "denoisr-chess-gui",
            "--checkpoint",
            "outputs/phase1.pt",
        ],
    )
    monkeypatch.setitem(
        sys.modules,
        "denoisr_chess.gui.app",
        types.SimpleNamespace(DenoisrApp=_DummyApp),
    )

    gui.main()

    app = _DummyApp.last_instance
    assert app is not None
    assert app._engine_mode_var.value == "single"
    assert app._ckpt_var.value == "outputs/phase1.pt"
    assert app.auto_started is True
    assert app.ran is True
