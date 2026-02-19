import pytest
import torch

from denoisr.nn.policy_backbone import ChessPolicyBackbone

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
            (n, p)
            for n, p in backbone.named_parameters()
            if "smolgen" in n
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
