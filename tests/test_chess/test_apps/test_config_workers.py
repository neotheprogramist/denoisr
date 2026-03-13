import logging

import pytest

import denoisr_chess.config as config


def test_resolve_dataloader_workers_auto_uses_conservative_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "_detect_available_cpus", lambda: 64)
    assert config.resolve_dataloader_workers(0) == 8


def test_resolve_dataloader_workers_auto_respects_low_cpu_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "_detect_available_cpus", lambda: 4)
    assert config.resolve_dataloader_workers(0) == 4


def test_resolve_dataloader_workers_clamps_explicit_requests(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(config, "_detect_available_cpus", lambda: 6)
    with caplog.at_level(logging.WARNING):
        assert config.resolve_dataloader_workers(12) == 6
    assert "clamping" in caplog.text


def test_resolve_dataloader_workers_keeps_valid_explicit_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(config, "_detect_available_cpus", lambda: 16)
    assert config.resolve_dataloader_workers(5) == 5
