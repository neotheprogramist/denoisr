"""Pipeline runner: orchestrates step execution with resume support."""

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Final

from denoisr_chess.pipeline.config import PipelineConfig
from denoisr_chess.pipeline.state import PipelineState
from denoisr_chess.pipeline.steps import (
    step_fetch_data,
    step_generate_data,
    step_init_model,
    step_train_phase1,
    step_train_phase2,
    step_train_phase3,
)

log = logging.getLogger(__name__)

ALL_STEPS = frozenset({"fetch", "init", "phase1", "phase2", "phase3"})
_PHASE_RANK: Final[dict[str, int]] = {
    "": 0,
    "init": 0,
    "fetched": 1,
    "model_initialized": 2,
    "phase1_complete": 3,
    "phase2_complete": 4,
    "phase3_complete": 5,
}


class PipelineRunner:
    """Runs the full training pipeline with resume and step-filtering support."""

    def __init__(
        self,
        cfg: PipelineConfig,
        restart: bool = False,
        only: frozenset[str] | None = None,
    ) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.output.dir)
        self.state_path = self.output_dir / "pipeline_state.json"
        self.only = only or ALL_STEPS

        if restart or not self.state_path.exists():
            self.state = PipelineState(
                started_at=datetime.now(timezone.utc).isoformat()
            )
        else:
            self.state = PipelineState.load(self.state_path)
            log.info("Resumed from state: phase=%s", self.state.phase)

    def _should_run(self, step: str) -> bool:
        return step in self.only

    def _save_state(self) -> None:
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state.save(self.state_path)

    def _run_timed_step(self, step_label: str, fn: Callable[[], None]) -> None:
        started = time.monotonic()
        log.info("%s", step_label)
        fn()
        elapsed = time.monotonic() - started
        log.info("%s complete (%.1fs)", step_label, elapsed)

    def _phase_rank(self) -> int:
        return _PHASE_RANK.get(self.state.phase, 0)

    def _run_single_step(
        self,
        *,
        step_key: str,
        step_label: str,
        min_completed_rank: int,
        fn: Callable[[], None],
    ) -> None:
        if not self._should_run(step_key):
            log.info("%s skipped (--only)", step_label)
            return
        if self._phase_rank() >= min_completed_rank:
            log.info("%s skipped (phase=%s)", step_label, self.state.phase)
            return
        self._run_timed_step(step_label, fn)
        self._save_state()

    def run(self) -> None:
        self._save_state()

        self._run_single_step(
            step_key="fetch",
            step_label="=== Step 1: Fetch data ===",
            min_completed_rank=1,
            fn=lambda: step_fetch_data(self.cfg, self.state),
        )

        self._run_single_step(
            step_key="init",
            step_label="=== Step 2: Initialize model ===",
            min_completed_rank=2,
            fn=lambda: step_init_model(self.cfg, self.state),
        )

        if not self._should_run("phase1"):
            log.info("=== Step 3: Generate training data === skipped (--only)")
            log.info("=== Step 4: Phase 1 training === skipped (--only)")
        elif self._phase_rank() >= 3:
            log.info(
                "=== Step 3: Generate training data === skipped (phase=%s)",
                self.state.phase,
            )
            log.info(
                "=== Step 4: Phase 1 training === skipped (phase=%s)",
                self.state.phase,
            )
        else:
            self._run_timed_step(
                "=== Step 3: Generate training data ===",
                lambda: step_generate_data(self.cfg, self.state),
            )
            self._save_state()
            self._run_timed_step(
                "=== Step 4: Phase 1 training ===",
                lambda: step_train_phase1(self.cfg, self.state),
            )
            self._save_state()

        self._run_single_step(
            step_key="phase2",
            step_label="=== Step 5: Phase 2: Diffusion bootstrapping ===",
            min_completed_rank=4,
            fn=lambda: step_train_phase2(self.cfg, self.state),
        )

        self._run_single_step(
            step_key="phase3",
            step_label="=== Step 6: Phase 3: RL self-play ===",
            min_completed_rank=5,
            fn=lambda: step_train_phase3(self.cfg, self.state),
        )

        log.info("Pipeline complete! Final checkpoint: %s", self.state.last_checkpoint)
