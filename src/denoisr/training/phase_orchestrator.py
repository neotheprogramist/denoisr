from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass(frozen=True)
class PhaseConfig:
    """Configuration for phase transition gates and alpha mixing.

    Gates:
    - Phase 1->2: top1_accuracy > phase1_gate (default 0.30)
    - Phase 2->3: diffusion_improvement_pp > phase2_gate (default 5.0)

    Alpha mixing (Phase 3):
    - alpha linearly increases from 0 to 1 over alpha_generations
    - final_policy = (1-alpha) * mcts_policy + alpha * diffusion_policy
    """

    phase1_gate: float = 0.30
    phase2_gate: float = 5.0
    alpha_generations: int = 50


class PhaseOrchestrator:
    """Manages training phase transitions and MCTS->diffusion alpha mixing."""

    def __init__(self, config: PhaseConfig) -> None:
        self._config = config
        self._phase = 1

    @property
    def current_phase(self) -> int:
        return self._phase

    def check_gate(self, metrics: dict[str, float]) -> bool:
        if self._phase == 1:
            if metrics.get("top1_accuracy", 0) > self._config.phase1_gate:
                self._phase = 2
                return True
        elif self._phase == 2:
            if (
                metrics.get("diffusion_improvement_pp", 0)
                > self._config.phase2_gate
            ):
                self._phase = 3
                return True
        return False

    def get_alpha(self, generation: int) -> float:
        if self._phase < 3:
            return 0.0
        return min(
            1.0, generation / max(1, self._config.alpha_generations)
        )

    def mix_policies(
        self, mcts_policy: Tensor, diffusion_policy: Tensor, alpha: float
    ) -> Tensor:
        return (1 - alpha) * mcts_policy + alpha * diffusion_policy
