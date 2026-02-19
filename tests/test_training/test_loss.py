import pytest
import torch

from denoisr.training.loss import ChessLossComputer


class TestChessLossComputer:
    @pytest.fixture
    def loss_fn(self) -> ChessLossComputer:
        return ChessLossComputer()

    def test_total_loss_is_scalar(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(4, 64, 64)
        pred_value = torch.softmax(torch.randn(4, 3), dim=-1)
        target_policy = torch.zeros(4, 64, 64)
        target_policy[:, 12, 28] = 1.0
        target_value = torch.tensor(
            [[1.0, 0.0, 0.0]] * 4, dtype=torch.float32
        )
        total, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert total.ndim == 0
        assert total.item() >= 0

    def test_breakdown_has_policy_and_value(
        self, loss_fn: ChessLossComputer
    ) -> None:
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 1] = 1.0
        target_value = torch.tensor([[0.0, 1.0, 0.0]] * 2)
        _, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert "policy" in breakdown
        assert "value" in breakdown
        assert all(v >= 0 for v in breakdown.values())

    def test_auxiliary_losses_included_in_total(
        self, loss_fn: ChessLossComputer
    ) -> None:
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        total_base, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        total_aux, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            consistency_loss=torch.tensor(0.5),
            diffusion_loss=torch.tensor(0.3),
            reward_loss=torch.tensor(0.1),
            ply_loss=torch.tensor(0.2),
        )
        assert total_aux.item() > total_base.item()
        assert "consistency" in breakdown
        assert "diffusion" in breakdown
        assert "reward" in breakdown
        assert "ply" in breakdown

    def test_all_6_terms_in_full_breakdown(self) -> None:
        loss_fn = ChessLossComputer()
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)
        _, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            consistency_loss=torch.tensor(0.5),
            diffusion_loss=torch.tensor(0.3),
            reward_loss=torch.tensor(0.1),
            ply_loss=torch.tensor(0.2),
        )
        expected_keys = {"policy", "value", "consistency", "diffusion", "reward", "ply", "total"}
        assert expected_keys == set(breakdown.keys())

    def test_perfect_prediction_low_loss(
        self, loss_fn: ChessLossComputer
    ) -> None:
        target_policy = torch.zeros(1, 64, 64)
        target_policy[0, 12, 28] = 1.0
        pred_policy = torch.full((1, 64, 64), -10.0)
        pred_policy[0, 12, 28] = 10.0

        target_value = torch.tensor([[1.0, 0.0, 0.0]])
        pred_value = torch.tensor([[0.95, 0.04, 0.01]])

        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert total.item() < 1.0

    def test_loss_is_finite(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(4, 64, 64)
        pred_value = torch.softmax(torch.randn(4, 3), dim=-1)
        target_policy = torch.zeros(4, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[0.5, 0.3, 0.2]] * 4)
        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        assert not torch.isnan(total)
        assert not torch.isinf(total)

    def test_gradient_flows(self, loss_fn: ChessLossComputer) -> None:
        pred_policy = torch.randn(2, 64, 64, requires_grad=True)
        pred_value = torch.softmax(
            torch.randn(2, 3, requires_grad=True), dim=-1
        )
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)
        total, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        total.backward()
        assert pred_policy.grad is not None

    def test_harmony_dream_adjusts_coefficients(self) -> None:
        loss_fn = ChessLossComputer(use_harmony_dream=True)
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        for _ in range(5):
            loss_fn.compute(
                pred_policy, pred_value, target_policy, target_value,
                consistency_loss=torch.tensor(10.0),
                diffusion_loss=torch.tensor(0.01),
            )
        coeffs = loss_fn.get_coefficients()
        assert "consistency" in coeffs
        assert "diffusion" in coeffs
