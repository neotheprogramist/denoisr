"""Tests for shared runtime helpers."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path

import pytest

from denoisr_common.runtime import configure_logging


def _reset_root_logging() -> None:
    logging.shutdown()
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)


def test_configure_logging_uses_timestamped_default(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("DENOISR_LOG_FILE", raising=False)

    try:
        path = configure_logging()
        resolved_path = path if path.is_absolute() else (tmp_path / path)
        assert resolved_path.parent == tmp_path / "logs"
        assert re.fullmatch(
            r"denoisr_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}\.log",
            resolved_path.name,
        )
        assert path == Path(os.environ["DENOISR_LOG_FILE"])
        logging.getLogger(__name__).info("runtime test line")
    finally:
        _reset_root_logging()

    assert resolved_path.exists()
    assert "runtime test line" in resolved_path.read_text(encoding="utf-8")


def test_configure_logging_respects_env_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    override = tmp_path / "custom.log"
    monkeypatch.setenv("DENOISR_LOG_FILE", str(override))

    try:
        path = configure_logging()
        assert path == override
    finally:
        _reset_root_logging()
