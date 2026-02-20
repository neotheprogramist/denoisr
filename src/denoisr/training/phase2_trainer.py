from dataclasses import dataclass

from torch import Tensor


@dataclass(frozen=True)
class TrajectoryBatch:
    """Enriched trajectory data for Phase 2 training.

    Each trajectory contains T consecutive board states connected by T-1 actions.
    Boards are raw encoder outputs, not BoardTensor newtypes.
    """

    boards: Tensor  # [N, T, C, 8, 8]
    actions_from: Tensor  # [N, T-1] (int64)
    actions_to: Tensor  # [N, T-1] (int64)
    policies: Tensor  # [N, T-1, 64, 64] one-hot from played move
    values: Tensor  # [N, 3] WDL from game result
    rewards: Tensor  # [N, T-1] per-move reward signal

    def __post_init__(self) -> None:
        B, T = self.boards.shape[:2]
        Tm1 = T - 1
        expected = {
            "actions_from": (B, Tm1),
            "actions_to": (B, Tm1),
            "policies": (B, Tm1, 64, 64),
            "rewards": (B, Tm1),
        }
        for name, shape in expected.items():
            actual = getattr(self, name).shape
            if actual[0] != B:
                raise ValueError(
                    f"{name} batch dim {actual[0]} != boards batch dim {B}"
                )
            if actual[1] != Tm1:
                raise ValueError(
                    f"{name} time dim {actual[1]} != expected {Tm1} "
                    f"(boards T={T})"
                )
        if self.values.shape[0] != B:
            raise ValueError(
                f"values batch dim {self.values.shape[0]} != boards batch "
                f"dim {B}"
            )
