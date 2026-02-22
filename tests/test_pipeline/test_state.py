from pathlib import Path
from denoisr.pipeline.state import PipelineState


def test_fresh_state() -> None:
    state = PipelineState()
    assert state.phase == "init"
    assert state.elo_tier_index == 0
    assert state.tier_accuracies == {}


def test_save_and_load(tmp_path: Path) -> None:
    state = PipelineState(phase="elo_curriculum", elo_tier_index=2)
    path = tmp_path / "state.json"
    state.save(path)
    loaded = PipelineState.load(path)
    assert loaded.phase == "elo_curriculum"
    assert loaded.elo_tier_index == 2


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


def test_tier_accuracies_round_trip(tmp_path: Path) -> None:
    state = PipelineState(
        phase="elo_curriculum",
        elo_tier_index=3,
        tier_accuracies={"800": 0.52, "1200": 0.51, "1600": 0.48},
    )
    path = tmp_path / "state.json"
    state.save(path)
    loaded = PipelineState.load(path)
    assert loaded.tier_accuracies == {"800": 0.52, "1200": 0.51, "1600": 0.48}


def test_load_ignores_unknown_fields(tmp_path: Path) -> None:
    """Forward compatibility: extra fields in JSON are ignored."""
    import json

    path = tmp_path / "state.json"
    data = {"phase": "sorted", "elo_tier_index": 0, "future_field": "value"}
    path.write_text(json.dumps(data))
    state = PipelineState.load(path)
    assert state.phase == "sorted"
