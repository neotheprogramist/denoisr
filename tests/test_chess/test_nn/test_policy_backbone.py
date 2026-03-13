import pytest
import torch

from denoisr_chess.nn.policy_backbone import ChessPolicyBackbone

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessPolicyBackbone:
    @pytest.fixture
    def backbone(self, device: torch.device) -> ChessPolicyBackbone:
        return ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)

    def test_output_shape_preserved(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        assert out.shape == small_latent.shape

    def test_gradient_flows_through_all_layers(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        out.sum().backward()
        for name, p in backbone.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"
            assert not torch.all(p.grad == 0), f"Zero gradient for {name}"

    def test_smolgen_biases_used(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        smolgen_params = [
            (n, p) for n, p in backbone.named_parameters() if "smolgen" in n
        ]
        assert len(smolgen_params) > 0
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        backbone(x).sum().backward()
        for name, p in smolgen_params:
            assert p.grad is not None, f"No gradient for {name}"

    def test_shaw_pe_biases_used(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        shaw_params = [
            (n, p)
            for n, p in backbone.named_parameters()
            if "shaw" in n or "relative" in n
        ]
        assert len(shaw_params) > 0
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        backbone(x).sum().backward()
        for name, p in shaw_params:
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan(
        self, backbone: ChessPolicyBackbone, small_latent: torch.Tensor
    ) -> None:
        out = backbone(small_latent)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_different_inputs_different_outputs(
        self, backbone: ChessPolicyBackbone, device: torch.device
    ) -> None:
        backbone.eval()
        x1 = torch.randn(1, 64, SMALL_D_S, device=device)
        x2 = torch.randn(1, 64, SMALL_D_S, device=device)
        assert not torch.allclose(backbone(x1), backbone(x2))

    def test_gradient_checkpointing_produces_gradients(
        self, device: torch.device
    ) -> None:
        """Backbone with gradient checkpointing should still produce valid gradients."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            gradient_checkpointing=True,
        ).to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        out = backbone(x)
        out.sum().backward()
        for name, p in backbone.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_gradient_checkpointing_matches_output(self, device: torch.device) -> None:
        """Checkpointed and non-checkpointed should produce identical forward output."""
        torch.manual_seed(42)
        bb_normal = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            gradient_checkpointing=False,
        ).to(device)
        torch.manual_seed(42)
        bb_ckpt = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            gradient_checkpointing=True,
        ).to(device)
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        assert torch.allclose(bb_normal(x), bb_ckpt(x), atol=1e-5)

    def test_sdpa_single_layer_valid(self, device: torch.device) -> None:
        """Single-layer backbone with SDPA produces finite correct-shape output."""
        torch.manual_seed(42)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=1,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        backbone.eval()
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        out = backbone(x)
        assert out.shape == (1, 64, SMALL_D_S)
        assert not torch.isnan(out).any()
        assert not torch.isinf(out).any()

    def test_dropout_params_accepted(self, device: torch.device) -> None:
        """TransformerBlock should accept dropout and drop_path_rate."""
        from denoisr_chess.nn.policy_backbone import TransformerBlock

        block = TransformerBlock(
            SMALL_D_S, SMALL_NUM_HEADS, SMALL_FFN_DIM,
            dropout=0.1, drop_path_rate=0.1,
        )
        block.to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        out = block(x)
        assert out.shape == x.shape

    def test_backbone_accepts_dropout(self, device: torch.device) -> None:
        """ChessPolicyBackbone should accept and distribute dropout/drop_path_rate."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            dropout=0.1,
            drop_path_rate=0.1,
        ).to(device)
        x = torch.randn(2, 64, SMALL_D_S, device=device)
        out = backbone(x)
        assert out.shape == x.shape

    def test_dropout_deterministic_in_non_training(self, device: torch.device) -> None:
        """Backbone should be deterministic in non-training mode even with dropout."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
            dropout=0.5,
            drop_path_rate=0.5,
        ).to(device)
        backbone.train(False)
        x = torch.randn(1, 64, SMALL_D_S, device=device)
        out1 = backbone(x)
        out2 = backbone(x)
        assert torch.allclose(out1, out2, atol=1e-6)

    def test_drop_path_linearly_scaled(self, device: torch.device) -> None:
        """DropPath rates should increase linearly across layers."""
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=4,
            ffn_dim=SMALL_FFN_DIM,
            drop_path_rate=0.3,
        ).to(device)
        rates = [layer.drop_path.drop_prob for layer in backbone.layers]
        assert rates[0] == 0.0
        assert abs(rates[-1] - 0.3) < 1e-6
        for i in range(len(rates) - 1):
            assert rates[i] <= rates[i + 1]
