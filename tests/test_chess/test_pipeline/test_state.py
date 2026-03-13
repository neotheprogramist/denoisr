from pathlib import Path

import pytest

from denoisr_chess.pipeline.state import PipelineState


def test_fresh_state() -> None:
    state = PipelineState()
    assert state.phase == "init"


def test_save_and_load(tmp_path: Path) -> None:
    state = PipelineState(phase="phase1_complete")
    path = tmp_path / "state.json"
    state.save(path)
    loaded = PipelineState.load(path)
    assert loaded.phase == "phase1_complete"


def test_load_missing_returns_fresh(tmp_path: Path) -> None:
    path = tmp_path / "nonexistent.json"
    state = PipelineState.load(path)
    assert state.phase == "init"


def test_save_creates_parent_dirs(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "state.json"
    state = PipelineState(phase="fetched")
    state.save(path)
    assert path.exists()
    loaded = PipelineState.load(path)
    assert loaded.phase == "fetched"


def test_load_ignores_unknown_fields(tmp_path: Path) -> None:
    """Forward compatibility: extra fields in JSON are ignored."""
    import json

    path = tmp_path / "state.json"
    data = {"phase": "fetched", "future_field": "value"}
    path.write_text(json.dumps(data))
    state = PipelineState.load(path)
    assert state.phase == "fetched"


def test_load_invalid_phase_fails_fast(tmp_path: Path) -> None:
    import json

    path = tmp_path / "state.json"
    path.write_text(json.dumps({"phase": "broken_state"}))
    with pytest.raises(ValueError, match="Invalid pipeline phase"):
        PipelineState.load(path)
