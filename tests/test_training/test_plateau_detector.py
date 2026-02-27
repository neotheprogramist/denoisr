import logging

from denoisr.training.plateau_detector import PlateauDetector


class TestPlateauDetector:
    def test_no_warnings_on_healthy_training(self) -> None:
        detector = PlateauDetector(window=3, grad_threshold=0.15)
        losses = [1.0, 0.8, 0.6, 0.4, 0.2]
        for epoch, loss in enumerate(losses):
            warnings = detector.update(epoch, grad_norm=1.0, loss=loss, lr=1e-3)
            assert warnings == []

    def test_warns_on_grad_norm_collapse(self) -> None:
        detector = PlateauDetector(grad_threshold=0.15)
        # Feed very small gradient norms to collapse EMA
        warnings: list[str] = []
        for epoch in range(20):
            w = detector.update(epoch, grad_norm=0.01, loss=1.0 - epoch * 0.01, lr=1e-3)
            warnings.extend(w)
        assert any("gradient norm EMA collapsed" in w for w in warnings)

    def test_warns_on_loss_stall(self) -> None:
        detector = PlateauDetector(window=5, loss_rel_threshold=1e-3)
        warnings: list[str] = []
        for epoch in range(10):
            # Loss barely changes
            w = detector.update(epoch, grad_norm=1.0, loss=1.0, lr=1e-3)
            warnings.extend(w)
        assert any("loss stalled" in w for w in warnings)

    def test_warns_on_low_effective_update(self) -> None:
        detector = PlateauDetector(update_mag_threshold=1e-4)
        # Very small LR × moderate grad norm
        warnings = detector.update(0, grad_norm=0.5, loss=1.0, lr=1e-5)
        assert any("effective update magnitude" in w for w in warnings)

    def test_logs_warnings(self, caplog: logging.LogCaptureFixture) -> None:
        detector = PlateauDetector(window=3, loss_rel_threshold=1e-3)
        with caplog.at_level(logging.WARNING):
            for epoch in range(5):
                detector.update(epoch, grad_norm=1.0, loss=1.0, lr=1e-3)
        assert any("loss stalled" in r.message for r in caplog.records)

    def test_suppresses_warnings_during_warmup(self) -> None:
        detector = PlateauDetector(warmup_epochs=3, update_mag_threshold=1e-2)
        warnings = [
            detector.update(epoch, grad_norm=0.01, loss=1.0, lr=1e-5)
            for epoch in range(3)
        ]
        assert warnings == [[], [], []]
