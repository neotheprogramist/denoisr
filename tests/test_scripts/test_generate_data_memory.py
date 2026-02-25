import logging

import pytest

from denoisr.scripts.generate_data import (
    _TenStepProgressTracker,
    _estimate_chunk_buffer_gib,
    _plan_chunk_examples_for_memory,
    _resolve_max_ram_gib,
)


def test_estimate_chunk_buffer_gib() -> None:
    gib = 1024 * 1024 * 1024
    num_examples = 1000
    num_planes = 122
    expected = num_examples * (((num_planes * 8 * 8) + (64 * 64) + 3) * 4) / gib
    assert _estimate_chunk_buffer_gib(num_examples, num_planes) == expected


def test_progress_tracker_logs_10_steps(caplog: pytest.LogCaptureFixture) -> None:
    tracker = _TenStepProgressTracker(total=100)
    with caplog.at_level(logging.INFO, logger="denoisr.scripts.generate_data"):
        for completed in range(1, 101):
            tracker.maybe_log(completed)

    progress_lines = [
        rec.message
        for rec in caplog.records
        if rec.message.startswith("Generation progress step")
    ]
    assert len(progress_lines) == 10
    assert "step 1/10 (10%): 10/100 examples" in progress_lines[0]
    assert "step 10/10 (100%): 100/100 examples" in progress_lines[-1]


def test_progress_tracker_validates_completed() -> None:
    tracker = _TenStepProgressTracker(total=10)
    with pytest.raises(ValueError, match="completed must be >= 0"):
        tracker.maybe_log(-1)


def test_plan_chunk_examples_caps_for_ram_budget() -> None:
    planned, est_peak_gib = _plan_chunk_examples_for_memory(
        requested_chunk_examples=1_000_000,
        max_ram_gib=64.0,
        num_planes=122,
        num_workers=64,
        chunksize=1024,
    )
    assert planned < 1_000_000
    assert est_peak_gib <= 64.0


def test_plan_chunk_examples_keeps_small_request() -> None:
    planned, est_peak_gib = _plan_chunk_examples_for_memory(
        requested_chunk_examples=2_048,
        max_ram_gib=64.0,
        num_planes=122,
        num_workers=4,
        chunksize=64,
    )
    assert planned == 2_048
    assert est_peak_gib > 0


def test_plan_chunk_examples_validates_inputs() -> None:
    with pytest.raises(ValueError, match="max_ram_gib must be > 0"):
        _plan_chunk_examples_for_memory(
            requested_chunk_examples=1,
            max_ram_gib=0.0,
            num_planes=122,
            num_workers=1,
            chunksize=1,
        )


def test_resolve_max_ram_gib_uses_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DENOISR_MAX_RAM_GIB", "72")
    assert _resolve_max_ram_gib(None) == 72.0


def test_resolve_max_ram_gib_rejects_invalid_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("DENOISR_MAX_RAM_GIB", "nope")
    with pytest.raises(ValueError, match="Invalid DENOISR_MAX_RAM_GIB"):
        _resolve_max_ram_gib(None)
