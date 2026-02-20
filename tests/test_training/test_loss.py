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
        assert breakdown["policy"] >= 0
        assert breakdown["value"] >= 0

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

    def test_illegal_logits_do_not_affect_loss(self, loss_fn: ChessLossComputer) -> None:
        """Changing logits at positions where target=0 should not change the loss."""
        B = 2
        pred_policy = torch.randn(B, 64, 64)
        pred_value = torch.softmax(torch.randn(B, 3), dim=-1)
        # Sparse target: only a few legal moves have nonzero probability
        target_policy = torch.zeros(B, 64, 64)
        target_policy[0, 4, 4] = 0.6  # e2-e4
        target_policy[0, 4, 12] = 0.4  # e2-e5
        target_policy[1, 1, 18] = 1.0  # single move
        target_value = torch.tensor([[0.4, 0.3, 0.3], [0.5, 0.2, 0.3]])

        loss_a, _ = loss_fn.compute(pred_policy, pred_value, target_policy, target_value)

        # Wildly change logits at illegal positions (where target is 0)
        pred_policy_b = pred_policy.clone()
        pred_policy_b[:, 0, 0] += 1000.0  # a1-a1 is never legal
        pred_policy_b[:, 7, 7] -= 1000.0

        loss_b, _ = loss_fn.compute(pred_policy_b, pred_value, target_policy, target_value)
        assert torch.allclose(loss_a, loss_b, atol=1e-5)

    def test_all_zero_target_does_not_produce_nan(
        self, loss_fn: ChessLossComputer
    ) -> None:
        """A batch item with no legal moves (all-zero target) must not produce NaN."""
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.softmax(torch.randn(2, 3), dim=-1)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[0, 4, 4] = 1.0  # item 0 has a legal move
        # item 1 has NO legal moves — simulates a corrupted/padded row
        target_value = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        total, _ = loss_fn.compute(pred_policy, pred_value, target_policy, target_value)
        assert not torch.isnan(total)
        assert not torch.isinf(total)

    def test_state_loss_included_in_total(self) -> None:
        loss_fn = ChessLossComputer()
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.randn(2, 3)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        total_base, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value
        )
        total_with_state, breakdown = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            state_loss=torch.tensor(0.5),
        )
        assert total_with_state.item() > total_base.item()
        assert "state" in breakdown

    def test_state_weight_scales_state_loss(self) -> None:
        loss_fn = ChessLossComputer(state_weight=2.0)
        pred_policy = torch.randn(2, 64, 64)
        pred_value = torch.randn(2, 3)
        target_policy = torch.zeros(2, 64, 64)
        target_policy[:, 0, 0] = 1.0
        target_value = torch.tensor([[1.0, 0.0, 0.0]] * 2)

        total_w1, _ = ChessLossComputer(state_weight=1.0).compute(
            pred_policy, pred_value, target_policy, target_value,
            state_loss=torch.tensor(1.0),
        )
        total_w2, _ = loss_fn.compute(
            pred_policy, pred_value, target_policy, target_value,
            state_loss=torch.tensor(1.0),
        )
        assert abs(total_w2.item() - total_w1.item() - 1.0) < 0.01

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
