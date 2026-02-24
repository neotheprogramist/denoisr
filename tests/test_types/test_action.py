import pytest
import torch
from hypothesis import given, strategies as st

from denoisr.types.action import Action, LegalMask


class TestAction:
    def test_valid(self) -> None:
        a = Action(from_square=0, to_square=63)
        assert a.from_square == 0 and a.to_square == 63

    def test_with_promotion(self) -> None:
        a = Action(from_square=52, to_square=60, promotion=5)
        assert a.promotion == 5

    @given(
        sq=st.integers(min_value=-100, max_value=-1)
        | st.integers(min_value=64, max_value=200)
    )
    def test_rejects_invalid_from_square(self, sq: int) -> None:
        with pytest.raises(ValueError, match="from_square"):
            Action(from_square=sq, to_square=0)

    @given(
        sq=st.integers(min_value=-100, max_value=-1)
        | st.integers(min_value=64, max_value=200)
    )
    def test_rejects_invalid_to_square(self, sq: int) -> None:
        with pytest.raises(ValueError, match="to_square"):
            Action(from_square=0, to_square=sq)

    def test_rejects_invalid_promotion(self) -> None:
        with pytest.raises(ValueError, match="promotion"):
            Action(from_square=0, to_square=0, promotion=1)

    def test_frozen(self) -> None:
        a = Action(0, 1)
        with pytest.raises(AttributeError):
            a.from_square = 5  # type: ignore[misc]


class TestLegalMask:
    def test_valid_shape(self) -> None:
        mask = LegalMask(torch.zeros(64, 64, dtype=torch.bool))
        assert mask.data.shape == (64, 64)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            LegalMask(torch.zeros(8, 8, dtype=torch.bool))

    def test_rejects_wrong_dtype(self) -> None:
        with pytest.raises(ValueError, match="bool"):
            LegalMask(torch.zeros(64, 64))
