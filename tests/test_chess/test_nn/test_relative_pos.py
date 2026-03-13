import pytest
import torch

from denoisr_chess.nn.relative_pos import ShawRelativePositionBias

from conftest import SMALL_NUM_HEADS


class TestShawRelativePositionBias:
    @pytest.fixture
    def pe(self) -> ShawRelativePositionBias:
        return ShawRelativePositionBias(num_heads=SMALL_NUM_HEADS)

    def test_output_shape(self, pe: ShawRelativePositionBias) -> None:
        out = pe()
        assert out.shape == (SMALL_NUM_HEADS, 64, 64)

    def test_topology_aware(self, pe: ShawRelativePositionBias) -> None:
        """Adjacent squares should have different bias than distant ones."""
        out = pe()
        # e2 (sq=12) to e4 (sq=28): rank diff=2, file diff=0
        # e2 (sq=12) to a8 (sq=56): rank diff=6, file diff=-4
        # These should have different biases
        assert not torch.allclose(out[:, 12, 28], out[:, 12, 56])

    def test_deterministic(self, pe: ShawRelativePositionBias) -> None:
        out1 = pe()
        out2 = pe()
        assert torch.equal(out1, out2)

    def test_gradient_flows(self, pe: ShawRelativePositionBias) -> None:
        out = pe()
        out.sum().backward()
        for p in pe.parameters():
            assert p.grad is not None
