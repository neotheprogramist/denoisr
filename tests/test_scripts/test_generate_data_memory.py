import logging

import pytest

from denoisr.scripts.generate_data import (
    _TenStepProgressTracker,
    _estimate_chunk_buffer_gib,
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
