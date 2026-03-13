"""Stratified holdout splitting for grokking detection.

Creates multiple independent holdout sets from TrainingExample lists:
- Random: baseline random split
- Game-level: entire games held out (no positional continuity leakage)
- Opening-family: entire ECO families held out (cross-structure generalization)
- Piece-count: endgame positions held out (phase-of-game generalization)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from denoisr_chess.types import TrainingExample


@dataclass(frozen=True)
class HoldoutSplits:
    train: list[TrainingExample]
    random: list[TrainingExample]
    game_level: list[TrainingExample]
    opening_family: list[TrainingExample]
    piece_count: list[TrainingExample]


class StratifiedHoldoutSplitter:
    def __init__(
        self,
        holdout_frac: float = 0.05,
        endgame_threshold: int = 6,
        seed: int = 42,
    ) -> None:
        self._holdout_frac = holdout_frac
        self._endgame_threshold = endgame_threshold
        self._rng = random.Random(seed)

    def split(self, examples: list[TrainingExample]) -> HoldoutSplits:
        n = len(examples)
        holdout_n = max(1, int(n * self._holdout_frac))
        excluded: set[int] = set()

        # 1. Piece-count holdout: all endgame positions
        piece_count_holdout: list[TrainingExample] = []
        for i, ex in enumerate(examples):
            if ex.piece_count is not None and ex.piece_count <= self._endgame_threshold:
                piece_count_holdout.append(ex)
                excluded.add(i)

        # 2. Game-level holdout: entire games
        game_level_holdout: list[TrainingExample] = []
        game_ids = {ex.game_id for ex in examples if ex.game_id is not None}
        if len(game_ids) >= 2:
            sorted_ids = sorted(game_ids)
            holdout_game_count = max(1, int(len(sorted_ids) * self._holdout_frac))
            holdout_game_count = min(holdout_game_count, len(sorted_ids) - 1)
            holdout_game_ids = set(self._rng.sample(sorted_ids, holdout_game_count))
            for i, ex in enumerate(examples):
                if ex.game_id in holdout_game_ids and i not in excluded:
                    game_level_holdout.append(ex)
                    excluded.add(i)

        # 3. Opening-family holdout: entire ECO letter groups
        opening_holdout: list[TrainingExample] = []
        eco_families = {ex.eco_code[0] for ex in examples if ex.eco_code}
        if len(eco_families) >= 2:
            holdout_family_count = max(1, int(len(eco_families) * self._holdout_frac))
            holdout_family_count = min(holdout_family_count, len(eco_families) - 1)
            holdout_families = set(
                self._rng.sample(sorted(eco_families), holdout_family_count)
            )
            for i, ex in enumerate(examples):
                if (
                    ex.eco_code
                    and ex.eco_code[0] in holdout_families
                    and i not in excluded
                ):
                    opening_holdout.append(ex)
                    excluded.add(i)

        # 4. Random holdout: from remaining examples
        remaining_indices = [i for i in range(n) if i not in excluded]
        random_holdout_n = min(holdout_n, len(remaining_indices))
        random_holdout_indices = set(
            self._rng.sample(remaining_indices, random_holdout_n)
        )
        random_holdout = [examples[i] for i in random_holdout_indices]
        excluded.update(random_holdout_indices)

        # 5. Train: everything not in any holdout
        train = [examples[i] for i in range(n) if i not in excluded]

        return HoldoutSplits(
            train=train,
            random=random_holdout,
            game_level=game_level_holdout,
            opening_family=opening_holdout,
            piece_count=piece_count_holdout,
        )
