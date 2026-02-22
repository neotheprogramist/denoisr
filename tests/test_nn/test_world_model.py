import pytest
import torch

from denoisr.nn.world_model import ChessWorldModel

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


class TestChessWorldModel:
    @pytest.fixture
    def model(self, device: torch.device) -> ChessWorldModel:
        return ChessWorldModel(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)

    def test_output_shapes(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        B, T = 2, 5
        states = torch.randn(B, T, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (B, T), device=device)
        act_to = torch.randint(0, 64, (B, T), device=device)
        next_states, rewards = model(states, act_from, act_to)
        assert next_states.shape == (B, T, 64, SMALL_D_S)
        assert rewards.shape == (B, T)

    def test_single_step(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(1, 1, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (1, 1), device=device)
        act_to = torch.randint(0, 64, (1, 1), device=device)
        next_states, rewards = model(states, act_from, act_to)
        assert next_states.shape == (1, 1, 64, SMALL_D_S)

    def test_causal_masking(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        """Changing future inputs should not affect past outputs."""
        B, T = 1, 4
        states = torch.randn(B, T, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (B, T), device=device)
        act_to = torch.randint(0, 64, (B, T), device=device)

        model.eval()
        out1, _ = model(states, act_from, act_to)

        states2 = states.clone()
        states2[:, -1] = torch.randn(1, 64, SMALL_D_S, device=device)
        out2, _ = model(states2, act_from, act_to)

        assert torch.allclose(out1[:, :-1], out2[:, :-1], atol=1e-5)

    def test_gradient_flows(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(2, 3, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (2, 3), device=device)
        act_to = torch.randint(0, 64, (2, 3), device=device)
        next_states, rewards = model(states, act_from, act_to)
        (next_states.sum() + rewards.sum()).backward()
        for name, p in model.named_parameters():
            assert p.grad is not None, f"No gradient for {name}"

    def test_no_nan(
        self, model: ChessWorldModel, device: torch.device
    ) -> None:
        states = torch.randn(2, 3, 64, SMALL_D_S, device=device)
        act_from = torch.randint(0, 64, (2, 3), device=device)
        act_to = torch.randint(0, 64, (2, 3), device=device)
        ns, rw = model(states, act_from, act_to)
        assert not torch.isnan(ns).any()
        assert not torch.isnan(rw).any()

    def test_has_set_attention_pool(
        self, model: ChessWorldModel
    ) -> None:
        """World model uses set attention pooling instead of mean pooling."""
        assert hasattr(model, "state_pool")
        assert hasattr(model.state_pool, "queries")
