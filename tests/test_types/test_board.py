import pytest
import torch
from hypothesis import given, strategies as st

from denoisr.types.board import BOARD_SIZE, NUM_PLANES, BoardTensor


class TestBoardTensor:
    def test_valid_shape(self) -> None:
        data = torch.zeros(NUM_PLANES, BOARD_SIZE, BOARD_SIZE)
        bt = BoardTensor(data)
        assert bt.data.shape == (NUM_PLANES, 8, 8)

    def test_rejects_wrong_ndim(self) -> None:
        with pytest.raises(ValueError, match="3D"):
            BoardTensor(torch.zeros(8, 8))

    def test_rejects_wrong_spatial(self) -> None:
        with pytest.raises(ValueError, match="8, 8"):
            BoardTensor(torch.zeros(12, 4, 4))

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="float32"):
            BoardTensor(torch.zeros(12, 8, 8, dtype=torch.int32))

    def test_frozen(self) -> None:
        bt = BoardTensor(torch.zeros(12, 8, 8))
        with pytest.raises(AttributeError):
            bt.data = torch.ones(12, 8, 8)  # type: ignore[misc]

    @given(c=st.integers(min_value=1, max_value=128))
    def test_accepts_any_channel_count(self, c: int) -> None:
        bt = BoardTensor(torch.zeros(c, 8, 8))
        assert bt.data.shape[0] == c
