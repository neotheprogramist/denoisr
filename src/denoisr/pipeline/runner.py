"""Pipeline runner: orchestrates step execution with resume support."""

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path

from denoisr.pipeline.config import PipelineConfig
from denoisr.pipeline.state import PipelineState
from denoisr.pipeline.steps import (
    step_fetch_data,
    step_generate_data,
    step_init_model,
    step_train_phase1,
    step_train_phase2,
    step_train_phase3,
)

log = logging.getLogger(__name__)

# Canonical step names for --only filtering
ALL_STEPS = frozenset({"fetch", "init", "phase1", "phase2", "phase3"})


class PipelineRunner:
    """Runs the full training pipeline with resume and step-filtering support.

    Takes a ``PipelineConfig``, an optional ``restart`` flag to discard
    previous state, and an optional ``only`` set to restrict which steps
    execute.  State is persisted to disk after every step so the pipeline
    can be resumed from any interruption point.
    """

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
        """Return whether *step* is included in the ``only`` filter."""
        return step in self.only

    def _save_state(self) -> None:
        """Persist current state to disk with an updated timestamp."""
        self.state.updated_at = datetime.now(timezone.utc).isoformat()
        self.state.save(self.state_path)

    def _run_timed_step(self, step_label: str, fn: Callable[[], None]) -> None:
        """Run one step and log elapsed wall time."""
        started = time.monotonic()
        log.info("%s", step_label)
        fn()
        elapsed = time.monotonic() - started
        log.info("%s complete (%.1fs)", step_label, elapsed)

    def run(self) -> None:
        """Execute the pipeline steps in sequence, skipping completed work."""
        self._save_state()

        if not self._should_run("fetch"):
            log.info("=== Step 1: Fetch data === skipped (--only)")
        elif self.state.phase not in ("init", ""):
            log.info("=== Step 1: Fetch data === skipped (phase=%s)", self.state.phase)
        else:
            self._run_timed_step(
                "=== Step 1: Fetch data ===",
                lambda: step_fetch_data(self.cfg, self.state),
            )
            self._save_state()

        if not self._should_run("init"):
            log.info("=== Step 2: Initialize model === skipped (--only)")
        elif self.state.phase not in (
            "fetched",
            "init",
            "",
        ):
            log.info(
                "=== Step 2: Initialize model === skipped (phase=%s)",
                self.state.phase,
            )
        else:
            self._run_timed_step(
                "=== Step 2: Initialize model ===",
                lambda: step_init_model(self.cfg, self.state),
            )
            self._save_state()

        if not self._should_run("phase1"):
            log.info("=== Step 3: Generate training data === skipped (--only)")
            log.info("=== Step 4: Phase 1 training === skipped (--only)")
        elif self.state.phase in (
            "phase1_complete",
            "phase2_complete",
            "phase3_complete",
        ):
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

        if not self._should_run("phase2"):
            log.info(
                "=== Step 5: Phase 2: Diffusion bootstrapping === skipped (--only)"
            )
        elif self.state.phase in (
            "phase2_complete",
            "phase3_complete",
        ):
            log.info(
                "=== Step 5: Phase 2: Diffusion bootstrapping === skipped (phase=%s)",
                self.state.phase,
            )
        else:
            self._run_timed_step(
                "=== Step 5: Phase 2: Diffusion bootstrapping ===",
                lambda: step_train_phase2(self.cfg, self.state),
            )
            self._save_state()

        if not self._should_run("phase3"):
            log.info("=== Step 6: Phase 3: RL self-play === skipped (--only)")
        elif self.state.phase == "phase3_complete":
            log.info("=== Step 6: Phase 3: RL self-play === skipped (already complete)")
        else:
            self._run_timed_step(
                "=== Step 6: Phase 3: RL self-play ===",
                lambda: step_train_phase3(self.cfg, self.state),
            )
            self._save_state()

        log.info("Pipeline complete! Final checkpoint: %s", self.state.last_checkpoint)
