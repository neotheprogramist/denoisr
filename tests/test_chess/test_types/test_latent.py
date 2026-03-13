import pytest
import torch

from denoisr_chess.types.latent import LatentState, LatentTrajectory


class TestLatentState:
    def test_valid(self) -> None:
        ls = LatentState(torch.randn(64, 128))
        assert ls.data.shape == (64, 128)

    def test_rejects_wrong_tokens(self) -> None:
        with pytest.raises(ValueError, match="64 tokens"):
            LatentState(torch.randn(32, 128))

    def test_rejects_1d(self) -> None:
        with pytest.raises(ValueError, match="2D"):
            LatentState(torch.randn(64))


class TestLatentTrajectory:
    def test_valid(self) -> None:
        states = tuple(LatentState(torch.randn(64, 128)) for _ in range(5))
        traj = LatentTrajectory(states)
        assert len(traj.states) == 5

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            LatentTrajectory(())

    def test_rejects_mismatched_d_s(self) -> None:
        s1 = LatentState(torch.randn(64, 128))
        s2 = LatentState(torch.randn(64, 256))
        with pytest.raises(ValueError, match="d_s"):
            LatentTrajectory((s1, s2))
