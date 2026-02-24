import pytest
import torch

from conftest import SMALL_D_S, SMALL_FFN_DIM, SMALL_NUM_HEADS, SMALL_NUM_LAYERS

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.grok_tracker import GrokState, GrokTracker


def _build_small_model(
    device: torch.device,
) -> tuple[ChessEncoder, ChessPolicyBackbone, ChessPolicyHead, ChessValueHead]:
    encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
    backbone = ChessPolicyBackbone(
        d_s=SMALL_D_S,
        num_heads=SMALL_NUM_HEADS,
        num_layers=SMALL_NUM_LAYERS,
        ffn_dim=SMALL_FFN_DIM,
    ).to(device)
    policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
    value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
    return encoder, backbone, policy_head, value_head


class TestGrokTrackerMetrics:
    def test_compute_weight_norms(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        norms = tracker.compute_weight_norms()
        assert "total" in norms
        assert "encoder" in norms
        assert "backbone" in norms
        assert "policy_head" in norms
        assert "value_head" in norms
        assert all(v > 0 for v in norms.values())
        tracker.close()

    def test_compute_effective_rank(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        # Run a forward pass to populate activation hooks
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        eranks = tracker.compute_effective_rank()
        assert len(eranks) == SMALL_NUM_LAYERS
        for erank in eranks.values():
            assert erank > 0
            assert erank <= SMALL_D_S  # Can't exceed embedding dimension
        tracker.close()

    def test_compute_spectral_norms(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        spectral = tracker.compute_spectral_norms()
        assert len(spectral) > 0
        assert all(v > 0 for v in spectral.values())
        tracker.close()

    def test_compute_htsr_alpha(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        alphas = tracker.compute_htsr_alpha()
        # With SMALL_D_S=64, the QKV weight is (192, 64) -> 64 singular values
        # 64 eigenvalues >= 10 threshold, so we should get alphas
        assert len(alphas) > 0
        for alpha in alphas.values():
            assert alpha > 0  # Power-law exponent must be positive
        tracker.close()

    def test_hooks_capture_activations(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        assert len(tracker._activations) == 0
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        assert len(tracker._activations) == SMALL_NUM_LAYERS
        tracker.close()

    def test_hooks_are_removed_on_close(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        tracker.close()
        tracker._activations.clear()
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        assert len(tracker._activations) == 0  # Hooks removed, no captures


class TestGrokTrackerStateMachine:
    def test_starts_in_baseline(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        assert tracker.state == GrokState.BASELINE
        tracker.close()

    def test_onset_detected_on_weight_norm_decrease(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            onset_threshold=0.95,
        )
        # Simulate weight norm history: first 50 at 10.0, last 50 at 9.0
        tracker._weight_norm_history = [10.0] * 50 + [9.0] * 50
        tracker._check_step_transitions(global_step=100)
        assert tracker.state == GrokState.ONSET_DETECTED
        tracker.close()

    def test_no_onset_if_norm_stable(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        tracker._weight_norm_history = [10.0] * 100
        tracker._check_step_transitions(global_step=100)
        assert tracker.state == GrokState.BASELINE
        tracker.close()

    def test_onset_increases_frequency(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        assert tracker._freq_multiplier == 1
        tracker._transition_to(GrokState.ONSET_DETECTED, 100, "test")
        assert tracker._freq_multiplier == 5
        tracker.close()

    def test_transitioning_increases_frequency_further(
        self, device: torch.device
    ) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        tracker._transition_to(GrokState.ONSET_DETECTED, 100, "test")
        tracker._transition_to(GrokState.TRANSITIONING, 200, "test")
        assert tracker._freq_multiplier == 10
        tracker.close()

    def test_step_returns_grok_state(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        # Run forward pass so hooks fire
        x = torch.randn(2, 12, 8, 8, device=device)
        latent = encoder(x)
        _ = backbone(latent)
        metrics = tracker.step(0, {"policy": 1.0, "value": 0.5, "total": 1.5}, 0.5)
        assert "grok/state" in metrics
        assert metrics["grok/state"] == 0.0  # BASELINE
        tracker.close()

    def test_epoch_computes_loss_gap(self, device: torch.device) -> None:
        encoder, backbone, policy_head, value_head = _build_small_model(device)
        tracker = GrokTracker(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
        )
        metrics = tracker.epoch(
            epoch=0,
            train_loss=5.0,
            holdout_metrics={
                "random": (0.01, 6.0),
                "game_level": (0.02, 5.5),
            },
        )
        assert "grok/loss_gap" in metrics
        assert metrics["grok/loss_gap"] == pytest.approx(5.0 - 5.5)
        tracker.close()
