import torch

from denoisr.training.threat_loss import ThreatHead, threat_loss

from conftest import SMALL_D_S


class TestThreatHead:
    def test_output_shape(self) -> None:
        head = ThreatHead(d_s=SMALL_D_S)
        features = torch.randn(4, 64, SMALL_D_S)
        logits = head(features)
        assert logits.shape == (4, 64)

    def test_gradient_flows(self) -> None:
        head = ThreatHead(d_s=SMALL_D_S)
        features = torch.randn(2, 64, SMALL_D_S, requires_grad=True)
        logits = head(features)
        logits.sum().backward()
        assert features.grad is not None
        assert head.linear.weight.grad is not None


class TestThreatLoss:
    def test_loss_is_finite(self) -> None:
        pred = torch.randn(4, 64)
        target = torch.zeros(4, 64)
        target[:, :10] = 1.0
        loss = threat_loss(pred, target)
        assert torch.isfinite(loss)

    def test_loss_decreases_on_correct_prediction(self) -> None:
        target = torch.zeros(4, 64)
        target[:, :10] = 1.0
        # Large positive logits where target=1, negative where target=0
        good_pred = torch.where(target > 0, torch.tensor(5.0), torch.tensor(-5.0))
        bad_pred = torch.where(target > 0, torch.tensor(-5.0), torch.tensor(5.0))
        assert threat_loss(good_pred, target) < threat_loss(bad_pred, target)
