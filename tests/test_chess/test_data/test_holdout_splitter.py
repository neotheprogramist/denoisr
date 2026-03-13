import torch

from denoisr_chess.data.holdout_splitter import HoldoutSplits, StratifiedHoldoutSplitter
from denoisr_chess.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget


def _make_example(
    game_id: int | None = None,
    eco_code: str | None = None,
    piece_count: int | None = None,
) -> TrainingExample:
    return TrainingExample(
        board=BoardTensor(torch.randn(12, 8, 8)),
        policy=PolicyTarget(torch.zeros(64, 64)),
        value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
        game_id=game_id,
        eco_code=eco_code,
        piece_count=piece_count,
    )


class TestStratifiedHoldoutSplitter:
    def test_returns_holdout_splits_dataclass(self) -> None:
        examples = [
            _make_example(game_id=i % 5, eco_code="B90", piece_count=30)
            for i in range(100)
        ]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        assert isinstance(splits, HoldoutSplits)
        assert len(splits.train) > 0
        assert len(splits.random) > 0

    def test_game_level_holdout_no_overlap(self) -> None:
        examples = [
            _make_example(game_id=i % 20, eco_code="B90", piece_count=30)
            for i in range(200)
        ]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        train_game_ids = {ex.game_id for ex in splits.train if ex.game_id is not None}
        holdout_game_ids = {
            ex.game_id for ex in splits.game_level if ex.game_id is not None
        }
        assert train_game_ids.isdisjoint(holdout_game_ids)

    def test_piece_count_holdout_all_endgame(self) -> None:
        examples = [
            _make_example(game_id=i, piece_count=p)
            for i, p in enumerate([32, 24, 16, 8, 5, 4, 3, 6, 2, 30])
        ]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        for ex in splits.piece_count:
            assert ex.piece_count is not None
            assert ex.piece_count <= 6

    def test_fallback_when_no_metadata(self) -> None:
        examples = [_make_example() for _ in range(100)]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.1, endgame_threshold=6)
        splits = splitter.split(examples)
        assert len(splits.random) > 0
        assert len(splits.game_level) == 0
        assert len(splits.opening_family) == 0
        assert len(splits.piece_count) == 0

    def test_train_does_not_contain_holdout_examples(self) -> None:
        examples = [
            _make_example(game_id=i % 10, eco_code="B90", piece_count=30)
            for i in range(100)
        ]
        splitter = StratifiedHoldoutSplitter(holdout_frac=0.2, endgame_threshold=6)
        splits = splitter.split(examples)
        all_holdout = {
            id(ex)
            for ex in splits.random
            + splits.game_level
            + splits.opening_family
            + splits.piece_count
        }
        train_ids = {id(ex) for ex in splits.train}
        assert all_holdout.isdisjoint(train_ids)
