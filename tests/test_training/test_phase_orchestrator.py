import pytest
import torch

from denoisr.training.phase_orchestrator import PhaseConfig, PhaseOrchestrator


class TestPhaseOrchestrator:
    @pytest.fixture
    def orchestrator(self) -> PhaseOrchestrator:
        return PhaseOrchestrator(PhaseConfig())

    def test_starts_at_phase_1(
        self, orchestrator: PhaseOrchestrator
    ) -> None:
        assert orchestrator.current_phase == 1

    def test_phase_1_to_2_gate(
        self, orchestrator: PhaseOrchestrator
    ) -> None:
        assert not orchestrator.check_gate(
            {"top1_accuracy": 0.25}
        )
        assert orchestrator.check_gate(
            {"top1_accuracy": 0.35}
        )
        assert orchestrator.current_phase == 2

    def test_phase_2_to_3_gate(self) -> None:
        o = PhaseOrchestrator(PhaseConfig())
        o.check_gate({"top1_accuracy": 0.35})
        assert o.current_phase == 2
        assert not o.check_gate(
            {"diffusion_improvement_pp": 3.0}
        )
        assert o.check_gate(
            {"diffusion_improvement_pp": 6.0}
        )
        assert o.current_phase == 3

    def test_alpha_mixing(self) -> None:
        o = PhaseOrchestrator(PhaseConfig(alpha_generations=10))
        o.check_gate({"top1_accuracy": 0.35})
        o.check_gate({"diffusion_improvement_pp": 6.0})
        assert o.current_phase == 3
        assert o.get_alpha(generation=0) == 0.0
        assert o.get_alpha(generation=5) == pytest.approx(0.5)
        assert o.get_alpha(generation=10) == pytest.approx(1.0)
        assert o.get_alpha(generation=20) == pytest.approx(1.0)

    def test_mixed_policy(self) -> None:
        o = PhaseOrchestrator(PhaseConfig())
        mcts_policy = torch.zeros(64, 64)
        mcts_policy[12, 28] = 1.0
        diff_policy = torch.zeros(64, 64)
        diff_policy[12, 20] = 1.0
        mixed = o.mix_policies(mcts_policy, diff_policy, alpha=0.5)
        assert mixed[12, 28].item() == pytest.approx(0.5)
        assert mixed[12, 20].item() == pytest.approx(0.5)
