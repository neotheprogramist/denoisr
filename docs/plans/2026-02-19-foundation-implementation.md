# Foundation Implementation Plan (Tiers 1–3)

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build types, game interface, and data pipeline — the foundation for the denoisr chess engine.

**Architecture:** Protocol-first design with frozen dataclasses. Data flows from Lichess PGN through encoders to PyTorch tensors. Each component tested independently via Protocols.

**Tech Stack:** Python 3.14, PyTorch (MPS), python-chess, zstandard, pytest, hypothesis

**Reference:** See `docs/plans/2026-02-19-component-decomposition-design.md` for the full architecture.

---

### Task 0: Project Scaffolding

**Files:**

- Modify: `pyproject.toml`
- Create: `src/denoisr/__init__.py` (empty)
- Create: `src/denoisr/types/__init__.py`
- Create: `src/denoisr/game/__init__.py` (empty)
- Create: `src/denoisr/data/__init__.py` (empty)
- Create: `src/denoisr/nn/__init__.py` (empty)
- Create: `src/denoisr/training/__init__.py` (empty)
- Create: `src/denoisr/inference/__init__.py` (empty)
- Create: `tests/conftest.py`

**Step 1: Add build system to pyproject.toml**

Add this section (hand-editing `[build-system]` is allowed — the CLAUDE.md rule only covers `[project.dependencies]`):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/denoisr"]
```

**Step 2: Add dependencies**

```bash
uv add torch chess zstandard
uv add --dev pytest hypothesis pytest-xdist
```

If `torch` fails on Python 3.14, fall back: `uv python pin 3.13 && uv add torch chess zstandard`.

**MPS fallback:** Set `PYTORCH_ENABLE_MPS_FALLBACK=1` in the development environment for graceful handling of unsupported MPS operations. Stick to float32/bfloat16 throughout (no float64 on MPS).

**Step 3: Create directory structure**

```bash
mkdir -p src/denoisr/{types,game,data,nn,training,inference}
mkdir -p tests/{test_types,test_game,test_data,test_nn,test_training,test_inference}
mkdir -p fixtures
```

Create empty `__init__.py` in every `src/denoisr/` subdirectory. Do NOT create `__init__.py` in test directories (pytest discovers tests without them).

**Step 4: Create `tests/conftest.py`**

```python
import chess
import pytest
import torch
from hypothesis import strategies as st


@pytest.fixture
def device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


SMALL_D_S = 64


@st.composite
def random_boards(draw: st.DrawFn) -> chess.Board:
    board = chess.Board()
    num_moves = draw(st.integers(min_value=0, max_value=60))
    for _ in range(num_moves):
        legal = list(board.legal_moves)
        if not legal:
            break
        move = draw(st.sampled_from(legal))
        board.push(move)
    return board
```

**Step 5: Verify setup**

Run: `uv run pytest --collect-only`
Expected: exits 0, no collection errors.

**Step 6: Commit**

```bash
git add pyproject.toml uv.lock src/ tests/conftest.py fixtures/
git commit -m "chore: scaffold project structure with deps and test infra"
```

---

### Task 1: Domain Types

**Files:**

- Create: `src/denoisr/types/board.py`
- Create: `src/denoisr/types/action.py`
- Create: `src/denoisr/types/latent.py`
- Create: `src/denoisr/types/training.py`
- Create: `src/denoisr/types/__init__.py` (re-exports)
- Test: `tests/test_types/test_board.py`
- Test: `tests/test_types/test_action.py`
- Test: `tests/test_types/test_latent.py`
- Test: `tests/test_types/test_training.py`

**Step 1: Write failing tests**

`tests/test_types/test_board.py`:

```python
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
```

`tests/test_types/test_action.py`:

```python
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

    @given(sq=st.integers(min_value=-100, max_value=-1) | st.integers(min_value=64, max_value=200))
    def test_rejects_invalid_from_square(self, sq: int) -> None:
        with pytest.raises(ValueError, match="from_square"):
            Action(from_square=sq, to_square=0)

    @given(sq=st.integers(min_value=-100, max_value=-1) | st.integers(min_value=64, max_value=200))
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
```

`tests/test_types/test_latent.py`:

```python
import pytest
import torch

from denoisr.types.latent import LatentState, LatentTrajectory


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
```

`tests/test_types/test_training.py`:

```python
import pytest
import torch

from denoisr.types.action import Action
from denoisr.types.board import BoardTensor
from denoisr.types.training import (
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


class TestPolicyTarget:
    def test_valid_shape(self) -> None:
        pt = PolicyTarget(torch.zeros(64, 64))
        assert pt.data.shape == (64, 64)

    def test_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError):
            PolicyTarget(torch.zeros(10, 10))


class TestValueTarget:
    def test_valid_wdl(self) -> None:
        vt = ValueTarget(win=0.7, draw=0.2, loss=0.1)
        assert abs(vt.win + vt.draw + vt.loss - 1.0) < 1e-5

    def test_rejects_bad_sum(self) -> None:
        with pytest.raises(ValueError, match="sum to 1.0"):
            ValueTarget(win=0.5, draw=0.5, loss=0.5)


class TestTrainingExample:
    def test_holds_components(self) -> None:
        board = BoardTensor(torch.zeros(12, 8, 8))
        policy = PolicyTarget(torch.zeros(64, 64))
        value = ValueTarget(1.0, 0.0, 0.0)
        ex = TrainingExample(board=board, policy=policy, value=value)
        assert ex.board is board


class TestGameRecord:
    def test_valid(self) -> None:
        actions = (Action(12, 28), Action(52, 36))
        gr = GameRecord(actions=actions, result=1.0)
        assert len(gr.actions) == 2

    def test_result_values(self) -> None:
        for r in (1.0, 0.0, -1.0):
            GameRecord(actions=(), result=r)
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_types/ -v`
Expected: ImportError — modules don't exist yet.

**Step 3: Implement types**

`src/denoisr/types/board.py`:

```python
from dataclasses import dataclass

import torch

BOARD_SIZE = 8
NUM_SQUARES = 64
NUM_PIECE_TYPES = 6
NUM_COLORS = 2
NUM_PLANES = NUM_PIECE_TYPES * NUM_COLORS


@dataclass(frozen=True)
class BoardTensor:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.ndim != 3:
            raise ValueError(f"Expected 3D tensor, got {self.data.ndim}D")
        if self.data.shape[1:] != (BOARD_SIZE, BOARD_SIZE):
            raise ValueError(
                f"Expected shape [C, 8, 8], got {list(self.data.shape)}"
            )
        if self.data.dtype != torch.float32:
            raise ValueError(f"Expected float32, got {self.data.dtype}")
```

`src/denoisr/types/action.py`:

```python
from dataclasses import dataclass

import torch

_VALID_PROMOTIONS = frozenset({2, 3, 4, 5})


@dataclass(frozen=True)
class Action:
    from_square: int
    to_square: int
    promotion: int | None = None

    def __post_init__(self) -> None:
        if not (0 <= self.from_square < 64):
            raise ValueError(
                f"from_square must be 0-63, got {self.from_square}"
            )
        if not (0 <= self.to_square < 64):
            raise ValueError(f"to_square must be 0-63, got {self.to_square}")
        if self.promotion is not None and self.promotion not in _VALID_PROMOTIONS:
            raise ValueError(
                f"promotion must be 2-5 or None, got {self.promotion}"
            )


@dataclass(frozen=True)
class LegalMask:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.shape != (64, 64):
            raise ValueError(
                f"Expected shape [64, 64], got {list(self.data.shape)}"
            )
        if self.data.dtype != torch.bool:
            raise ValueError(f"Expected bool, got {self.data.dtype}")
```

`src/denoisr/types/latent.py`:

```python
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LatentState:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.ndim != 2:
            raise ValueError(f"Expected 2D tensor, got {self.data.ndim}D")
        if self.data.shape[0] != 64:
            raise ValueError(
                f"Expected 64 tokens, got {self.data.shape[0]}"
            )


@dataclass(frozen=True)
class LatentTrajectory:
    states: tuple[LatentState, ...]

    def __post_init__(self) -> None:
        if len(self.states) == 0:
            raise ValueError("Trajectory must have at least one state")
        d_s = self.states[0].data.shape[1]
        for i, s in enumerate(self.states):
            if s.data.shape[1] != d_s:
                raise ValueError(
                    f"State {i} has d_s={s.data.shape[1]}, expected {d_s}"
                )
```

`src/denoisr/types/training.py`:

```python
from dataclasses import dataclass

import torch

from denoisr.types.action import Action
from denoisr.types.board import BoardTensor


@dataclass(frozen=True)
class PolicyTarget:
    data: torch.Tensor

    def __post_init__(self) -> None:
        if self.data.shape != (64, 64):
            raise ValueError(
                f"Expected shape [64, 64], got {list(self.data.shape)}"
            )


@dataclass(frozen=True)
class ValueTarget:
    win: float
    draw: float
    loss: float

    def __post_init__(self) -> None:
        total = self.win + self.draw + self.loss
        if abs(total - 1.0) > 1e-5:
            raise ValueError(f"WDL must sum to 1.0, got {total}")


@dataclass(frozen=True)
class TrainingExample:
    board: BoardTensor
    policy: PolicyTarget
    value: ValueTarget


@dataclass(frozen=True)
class GameRecord:
    actions: tuple[Action, ...]
    result: float
```

`src/denoisr/types/__init__.py`:

```python
from denoisr.types.action import Action, LegalMask
from denoisr.types.board import BOARD_SIZE, NUM_PLANES, NUM_SQUARES, BoardTensor
from denoisr.types.latent import LatentState, LatentTrajectory
from denoisr.types.training import (
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)

__all__ = [
    "Action",
    "BOARD_SIZE",
    "BoardTensor",
    "GameRecord",
    "LatentState",
    "LatentTrajectory",
    "LegalMask",
    "NUM_PLANES",
    "NUM_SQUARES",
    "PolicyTarget",
    "TrainingExample",
    "ValueTarget",
]
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_types/ -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/types/ tests/test_types/
git commit -m "feat: add domain types with validation (Layer 0)"
```

---

### Task 2: Game Interface

**Files:**

- Create: `src/denoisr/game/protocols.py`
- Create: `src/denoisr/game/chess_game.py`
- Test: `tests/test_game/test_chess_game.py`

**Step 1: Write protocol**

`src/denoisr/game/protocols.py`:

```python
from typing import Protocol

import chess
import torch

from denoisr.types import Action, LegalMask


class GameInterface(Protocol):
    def get_init_board(self) -> chess.Board: ...
    def get_board_size(self) -> tuple[int, int]: ...
    def get_action_size(self) -> int: ...
    def get_next_state(
        self, board: chess.Board, action: Action
    ) -> chess.Board: ...
    def get_valid_moves(self, board: chess.Board) -> LegalMask: ...
    def get_game_ended(self, board: chess.Board) -> float | None: ...
    def get_canonical_form(self, board: chess.Board) -> chess.Board: ...
    def get_symmetries(
        self, board: chess.Board, policy: torch.Tensor
    ) -> list[tuple[chess.Board, torch.Tensor]]: ...
```

**Step 2: Write failing tests**

`tests/test_game/test_chess_game.py`:

```python
import chess
import pytest
import torch
from hypothesis import given, settings

from denoisr.game.chess_game import ChessGame
from denoisr.types import Action

from tests.conftest import random_boards


class TestChessGame:
    @pytest.fixture
    def game(self) -> ChessGame:
        return ChessGame()

    def test_init_board_is_starting_position(self, game: ChessGame) -> None:
        board = game.get_init_board()
        assert board.fen() == chess.STARTING_FEN

    def test_board_size(self, game: ChessGame) -> None:
        assert game.get_board_size() == (8, 8)

    def test_action_size(self, game: ChessGame) -> None:
        assert game.get_action_size() == 64 * 64

    def test_next_state_applies_move(self, game: ChessGame) -> None:
        board = game.get_init_board()
        action = Action(from_square=12, to_square=28)  # e2e4
        new_board = game.get_next_state(board, action)
        assert new_board.piece_at(28) == chess.Piece(chess.PAWN, chess.WHITE)
        assert new_board.piece_at(12) is None

    def test_next_state_does_not_mutate_original(
        self, game: ChessGame
    ) -> None:
        board = game.get_init_board()
        original_fen = board.fen()
        game.get_next_state(board, Action(12, 28))
        assert board.fen() == original_fen

    def test_valid_moves_starting_position(self, game: ChessGame) -> None:
        board = game.get_init_board()
        mask = game.get_valid_moves(board)
        assert mask.data.shape == (64, 64)
        assert mask.data.dtype == torch.bool
        num_legal = len(list(board.legal_moves))
        assert mask.data.sum().item() == num_legal

    def test_game_ended_none_at_start(self, game: ChessGame) -> None:
        assert game.get_game_ended(game.get_init_board()) is None

    def test_game_ended_scholars_mate(self, game: ChessGame) -> None:
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push_uci(uci)
        assert game.get_game_ended(board) == 1.0

    def test_game_ended_fools_mate(self, game: ChessGame) -> None:
        board = chess.Board()
        for uci in ("f2f3", "e7e5", "g2g4", "d8h4"):
            board.push_uci(uci)
        assert game.get_game_ended(board) == -1.0

    def test_canonical_form_white_unchanged(self, game: ChessGame) -> None:
        board = game.get_init_board()
        canonical = game.get_canonical_form(board)
        assert canonical.fen() == board.fen()

    def test_canonical_form_black_mirrored(self, game: ChessGame) -> None:
        board = chess.Board()
        board.push_uci("e2e4")  # now black to move
        canonical = game.get_canonical_form(board)
        assert canonical.turn == chess.WHITE

    def test_symmetries_returns_identity(self, game: ChessGame) -> None:
        board = game.get_init_board()
        policy = torch.randn(64, 64)
        syms = game.get_symmetries(board, policy)
        assert len(syms) == 1
        assert syms[0][0].fen() == board.fen()
        assert torch.equal(syms[0][1], policy)

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_valid_moves_count_matches_python_chess(
        self, game: ChessGame, board: chess.Board
    ) -> None:
        if game.get_game_ended(board) is not None:
            return
        mask = game.get_valid_moves(board)
        expected = len(list(board.legal_moves))
        assert mask.data.sum().item() == expected
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_game/ -v`
Expected: ImportError.

**Step 4: Implement**

`src/denoisr/game/chess_game.py`:

```python
import chess
import torch

from denoisr.types import Action, LegalMask


class ChessGame:
    def get_init_board(self) -> chess.Board:
        return chess.Board()

    def get_board_size(self) -> tuple[int, int]:
        return (8, 8)

    def get_action_size(self) -> int:
        return 64 * 64

    def get_next_state(
        self, board: chess.Board, action: Action
    ) -> chess.Board:
        new_board = board.copy()
        move = chess.Move(
            action.from_square, action.to_square, action.promotion
        )
        new_board.push(move)
        return new_board

    def get_valid_moves(self, board: chess.Board) -> LegalMask:
        mask = torch.zeros(64, 64, dtype=torch.bool)
        for move in board.legal_moves:
            mask[move.from_square, move.to_square] = True
        return LegalMask(mask)

    def get_game_ended(self, board: chess.Board) -> float | None:
        if not board.is_game_over():
            return None
        result = board.result()
        if result == "1-0":
            return 1.0
        if result == "0-1":
            return -1.0
        return 0.0

    def get_canonical_form(self, board: chess.Board) -> chess.Board:
        if board.turn == chess.BLACK:
            return board.mirror()
        return board.copy()

    def get_symmetries(
        self, board: chess.Board, policy: torch.Tensor
    ) -> list[tuple[chess.Board, torch.Tensor]]:
        return [(board.copy(), policy.clone())]
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_game/ -v`
Expected: All pass.

**Step 6: Commit**

```bash
git add src/denoisr/game/ tests/test_game/
git commit -m "feat: add game interface with chess implementation (Layer 1)"
```

---

### Task 3: Data Protocols + Board Encoder

**Files:**

- Create: `src/denoisr/data/protocols.py`
- Create: `src/denoisr/data/board_encoder.py`
- Test: `tests/test_data/test_board_encoder.py`

**Step 1: Write data protocols**

`src/denoisr/data/protocols.py`:

```python
from collections.abc import Iterator
from pathlib import Path
from typing import Protocol

import chess

from denoisr.types import (
    Action,
    BoardTensor,
    GameRecord,
    PolicyTarget,
    ValueTarget,
)


class BoardEncoder(Protocol):
    @property
    def num_planes(self) -> int: ...
    def encode(self, board: chess.Board) -> BoardTensor: ...


class ActionEncoder(Protocol):
    def encode_move(self, move: chess.Move) -> Action: ...
    def decode_action(
        self, action: Action, board: chess.Board
    ) -> chess.Move: ...
    def action_to_index(self, action: Action) -> int: ...
    def index_to_action(self, index: int, board: chess.Board) -> Action: ...


class PGNStreamer(Protocol):
    def stream(self, path: Path) -> Iterator[GameRecord]: ...


class Oracle(Protocol):
    def evaluate(
        self, board: chess.Board
    ) -> tuple[PolicyTarget, ValueTarget, float]: ...
```

**Step 2: Write failing tests**

`tests/test_data/test_board_encoder.py`:

```python
import chess
import torch
from hypothesis import given, settings

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.types import BoardTensor

from tests.conftest import random_boards


class TestSimpleBoardEncoder:
    @pytest.fixture
    def encoder(self) -> SimpleBoardEncoder:
        return SimpleBoardEncoder()

    def test_num_planes(self, encoder: SimpleBoardEncoder) -> None:
        assert encoder.num_planes == 12

    def test_starting_position_shape(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.shape == (12, 8, 8)

    def test_starting_position_white_pawns(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        # Plane 0 = white pawns, rank 1 (index 1) = all pawns
        assert bt.data[0, 1, :].sum().item() == 8
        assert bt.data[0, 1, :].all()

    def test_starting_position_white_king(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        # Plane 5 = white king, e1 = rank 0, file 4
        assert bt.data[5, 0, 4].item() == 1.0
        assert bt.data[5].sum().item() == 1.0

    def test_starting_position_total_pieces(
        self, encoder: SimpleBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.sum().item() == 32  # 16 white + 16 black pieces

    def test_empty_board(self, encoder: SimpleBoardEncoder) -> None:
        board = chess.Board.empty()
        bt = encoder.encode(board)
        assert bt.data.sum().item() == 0.0

    def test_deterministic(self, encoder: SimpleBoardEncoder) -> None:
        board = chess.Board()
        bt1 = encoder.encode(board)
        bt2 = encoder.encode(board)
        assert torch.equal(bt1.data, bt2.data)

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_piece_count_matches(
        self, encoder: SimpleBoardEncoder, board: chess.Board
    ) -> None:
        bt = encoder.encode(board)
        total_tensor_pieces = bt.data.sum().item()
        total_board_pieces = len(board.piece_map())
        assert total_tensor_pieces == total_board_pieces

    @given(board=random_boards())
    @settings(max_examples=50)
    def test_no_overlapping_pieces(
        self, encoder: SimpleBoardEncoder, board: chess.Board
    ) -> None:
        bt = encoder.encode(board)
        # Sum across all 12 planes — no square should have more than 1
        per_square = bt.data.sum(dim=0)
        assert per_square.max().item() <= 1.0
```

Note: add `import pytest` at the top of the test file.

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_board_encoder.py -v`
Expected: ImportError.

**Step 4: Implement**

`src/denoisr/data/board_encoder.py`:

```python
import chess
import torch

from denoisr.types import BoardTensor

_PLANE_INDEX = {
    (pt, color): (pt - 1) + (0 if color == chess.WHITE else 6)
    for pt in chess.PIECE_TYPES
    for color in chess.COLORS
}


class SimpleBoardEncoder:
    @property
    def num_planes(self) -> int:
        return 12

    def encode(self, board: chess.Board) -> BoardTensor:
        data = torch.zeros(12, 8, 8, dtype=torch.float32)
        for sq, piece in board.piece_map().items():
            plane = _PLANE_INDEX[(piece.piece_type, piece.color)]
            rank = chess.square_rank(sq)
            file = chess.square_file(sq)
            data[plane, rank, file] = 1.0
        return BoardTensor(data)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_data/test_board_encoder.py -v`
Expected: All pass.

**Step 6: Commit**

```bash
git add src/denoisr/data/ tests/test_data/test_board_encoder.py
git commit -m "feat: add data protocols and board encoder (Layer 2a)"
```

---

### Task 4: Action Encoder

**Files:**

- Create: `src/denoisr/data/action_encoder.py`
- Test: `tests/test_data/test_action_encoder.py`

**Step 1: Write failing tests**

`tests/test_data/test_action_encoder.py`:

```python
import chess
import pytest
from hypothesis import given, settings

from denoisr.data.action_encoder import SimpleActionEncoder
from denoisr.types import Action

from tests.conftest import random_boards


class TestSimpleActionEncoder:
    @pytest.fixture
    def encoder(self) -> SimpleActionEncoder:
        return SimpleActionEncoder()

    def test_encode_e2e4(self, encoder: SimpleActionEncoder) -> None:
        move = chess.Move.from_uci("e2e4")
        action = encoder.encode_move(move)
        assert action.from_square == chess.E2
        assert action.to_square == chess.E4
        assert action.promotion is None

    def test_encode_promotion(self, encoder: SimpleActionEncoder) -> None:
        move = chess.Move.from_uci("a7a8q")
        action = encoder.encode_move(move)
        assert action.promotion == chess.QUEEN

    def test_decode_e2e4(self, encoder: SimpleActionEncoder) -> None:
        board = chess.Board()
        action = Action(chess.E2, chess.E4)
        move = encoder.decode_action(action, board)
        assert move == chess.Move.from_uci("e2e4")

    def test_round_trip_starting_position(
        self, encoder: SimpleActionEncoder
    ) -> None:
        board = chess.Board()
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            decoded = encoder.decode_action(action, board)
            assert decoded == move

    def test_action_to_index_range(
        self, encoder: SimpleActionEncoder
    ) -> None:
        action = Action(0, 63)
        idx = encoder.action_to_index(action)
        assert 0 <= idx < 64 * 64

    def test_index_round_trip(self, encoder: SimpleActionEncoder) -> None:
        board = chess.Board()
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            idx = encoder.action_to_index(action)
            recovered = encoder.index_to_action(idx, board)
            decoded = encoder.decode_action(recovered, board)
            assert decoded == move

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_all_legal_moves_round_trip(
        self, encoder: SimpleActionEncoder, board: chess.Board
    ) -> None:
        for move in board.legal_moves:
            action = encoder.encode_move(move)
            idx = encoder.action_to_index(action)
            assert 0 <= idx < 64 * 64
            recovered = encoder.index_to_action(idx, board)
            decoded = encoder.decode_action(recovered, board)
            # For underpromotions, we default to queen — skip exact match
            if move.promotion and move.promotion != chess.QUEEN:
                continue
            assert decoded == move
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_action_encoder.py -v`
Expected: ImportError.

**Step 3: Implement**

`src/denoisr/data/action_encoder.py`:

```python
import chess

from denoisr.types import Action


class SimpleActionEncoder:
    def encode_move(self, move: chess.Move) -> Action:
        return Action(move.from_square, move.to_square, move.promotion)

    def decode_action(self, action: Action, board: chess.Board) -> chess.Move:
        return chess.Move(
            action.from_square, action.to_square, action.promotion
        )

    def action_to_index(self, action: Action) -> int:
        return action.from_square * 64 + action.to_square

    def index_to_action(self, index: int, board: chess.Board) -> Action:
        from_sq = index // 64
        to_sq = index % 64
        promotion = self._infer_promotion(from_sq, to_sq, board)
        return Action(from_sq, to_sq, promotion)

    def _infer_promotion(
        self, from_sq: int, to_sq: int, board: chess.Board
    ) -> int | None:
        piece = board.piece_at(from_sq)
        if piece is None or piece.piece_type != chess.PAWN:
            return None
        to_rank = chess.square_rank(to_sq)
        if piece.color == chess.WHITE and to_rank == 7:
            return chess.QUEEN
        if piece.color == chess.BLACK and to_rank == 0:
            return chess.QUEEN
        return None
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data/test_action_encoder.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/data/action_encoder.py tests/test_data/test_action_encoder.py
git commit -m "feat: add action encoder with index mapping (Layer 2b)"
```

---

### Task 5: PGN Streamer

**Scaling notes:** The Lichess open database (database.lichess.org, CC0) contains 7.2B+ games in monthly `.pgn.zst` files (~4GB compressed → ~28GB decompressed). Full python-chess parsing runs at ~15,000 games/minute — for scale, use `chess.pgn.scan_headers()` to pre-filter by rating (e.g. 2000+) before full parsing. The **Lichess Elite Database** (database.nikonoel.fr) pre-filters for 2400+ rated players, dramatically reducing volume. ~6% of Lichess games include embedded Stockfish evaluations as PGN comments (`[%eval 2.35]`).

**Files:**

- Create: `fixtures/sample_games.pgn`
- Create: `src/denoisr/data/pgn_streamer.py`
- Test: `tests/test_data/test_pgn_streamer.py`

**Step 1: Create test fixture**

`fixtures/sample_games.pgn`:

```pgn
[Event "Scholar's Mate"]
[Result "1-0"]

1. e4 e5 2. Qh5 Nc6 3. Bc4 Nf6 4. Qxf7# 1-0

[Event "Fool's Mate"]
[Result "0-1"]

1. f3 e5 2. g4 Qh4# 0-1

[Event "Short Draw"]
[Result "1/2-1/2"]

1. e4 e5 2. Nf3 Nf6 3. Nxe5 Nxe4 1/2-1/2
```

**Step 2: Write failing tests**

`tests/test_data/test_pgn_streamer.py`:

```python
from pathlib import Path

import pytest

from denoisr.data.pgn_streamer import SimplePGNStreamer
from denoisr.types import GameRecord

FIXTURES = Path(__file__).resolve().parent.parent.parent / "fixtures"


class TestSimplePGNStreamer:
    @pytest.fixture
    def streamer(self) -> SimplePGNStreamer:
        return SimplePGNStreamer()

    def test_streams_correct_count(
        self, streamer: SimplePGNStreamer
    ) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        assert len(games) == 3

    def test_scholars_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[0]
        assert game.result == 1.0
        assert len(game.actions) == 7  # 4 white + 3 black moves

    def test_fools_mate(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[1]
        assert game.result == -1.0
        assert len(game.actions) == 4

    def test_draw(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        game = games[2]
        assert game.result == 0.0

    def test_actions_are_valid(self, streamer: SimplePGNStreamer) -> None:
        games = list(streamer.stream(FIXTURES / "sample_games.pgn"))
        for game in games:
            for action in game.actions:
                assert 0 <= action.from_square < 64
                assert 0 <= action.to_square < 64

    def test_empty_file(
        self, streamer: SimplePGNStreamer, tmp_path: Path
    ) -> None:
        empty = tmp_path / "empty.pgn"
        empty.write_text("")
        games = list(streamer.stream(empty))
        assert len(games) == 0
```

**Step 3: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_pgn_streamer.py -v`
Expected: ImportError.

**Step 4: Implement**

`src/denoisr/data/pgn_streamer.py`:

```python
import io
from collections.abc import Iterator
from pathlib import Path

import chess.pgn

from denoisr.types import Action, GameRecord

_RESULT_MAP = {"1-0": 1.0, "0-1": -1.0, "1/2-1/2": 0.0}


class SimplePGNStreamer:
    def stream(self, path: Path) -> Iterator[GameRecord]:
        suffix = "".join(path.suffixes)
        if suffix.endswith(".zst"):
            yield from self._stream_zst(path)
        else:
            yield from self._stream_pgn(path)

    def _stream_pgn(self, path: Path) -> Iterator[GameRecord]:
        with open(path) as f:
            yield from self._parse_games(f)

    def _stream_zst(self, path: Path) -> Iterator[GameRecord]:
        import zstandard as zstd

        with open(path, "rb") as fh:
            reader = zstd.ZstdDecompressor().stream_reader(fh)
            text_stream = io.TextIOWrapper(reader, encoding="utf-8")
            yield from self._parse_games(text_stream)

    def _parse_games(
        self, stream: io.TextIOBase
    ) -> Iterator[GameRecord]:
        while True:
            game = chess.pgn.read_game(stream)
            if game is None:
                break
            result_str = game.headers.get("Result", "*")
            result = _RESULT_MAP.get(result_str)
            if result is None:
                continue
            actions = tuple(
                Action(m.from_square, m.to_square, m.promotion)
                for m in game.mainline_moves()
            )
            yield GameRecord(actions=actions, result=result)
```

**Step 5: Run tests**

Run: `uv run pytest tests/test_data/test_pgn_streamer.py -v`
Expected: All pass.

**Step 6: Commit**

```bash
git add fixtures/sample_games.pgn src/denoisr/data/pgn_streamer.py tests/test_data/test_pgn_streamer.py
git commit -m "feat: add PGN streamer with zstd support (Layer 2c)"
```

---

### Task 6: Stockfish Oracle

**Scaling notes:** For large-scale evaluation, spawn multiple single-threaded Stockfish instances via `multiprocessing.Pool`, each evaluating positions through `chess.engine`. Configure each instance with `Threads=1` and run N instances across N CPU cores. At depth 10–12, expect ~100–500 positions/second per thread. The async API (`chess.engine.popen_uci` with `await`) enables concurrent evaluation for even higher throughput.

**Files:**

- Create: `src/denoisr/data/stockfish_oracle.py`
- Test: `tests/test_data/test_stockfish_oracle.py`

**Step 1: Write failing tests**

`tests/test_data/test_stockfish_oracle.py`:

```python
import shutil

import chess
import pytest

from denoisr.data.stockfish_oracle import StockfishOracle

STOCKFISH_PATH = shutil.which("stockfish")
pytestmark = pytest.mark.skipif(
    STOCKFISH_PATH is None, reason="stockfish not installed"
)


class TestStockfishOracle:
    @pytest.fixture
    def oracle(self) -> StockfishOracle:
        assert STOCKFISH_PATH is not None
        return StockfishOracle(path=STOCKFISH_PATH, depth=10)

    def test_starting_position_policy_is_distribution(
        self, oracle: StockfishOracle
    ) -> None:
        policy, _, _ = oracle.evaluate(chess.Board())
        total = policy.data.sum().item()
        assert abs(total - 1.0) < 0.01

    def test_starting_position_value_is_wdl(
        self, oracle: StockfishOracle
    ) -> None:
        _, value, _ = oracle.evaluate(chess.Board())
        assert abs(value.win + value.draw + value.loss - 1.0) < 1e-5

    def test_eval_is_finite(self, oracle: StockfishOracle) -> None:
        _, _, cp = oracle.evaluate(chess.Board())
        assert -10000 <= cp <= 10000

    def test_mate_position_value(self, oracle: StockfishOracle) -> None:
        # Position after Scholar's mate — white already won
        board = chess.Board()
        for uci in ("e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"):
            board.push_uci(uci)
        # Game is over, but oracle should still handle it
        # (implementation may return extreme values)

    def test_policy_only_on_legal_moves(
        self, oracle: StockfishOracle
    ) -> None:
        board = chess.Board()
        policy, _, _ = oracle.evaluate(board)
        # Check that nonzero entries correspond to legal moves
        for from_sq in range(64):
            for to_sq in range(64):
                if policy.data[from_sq, to_sq].item() > 0:
                    found = any(
                        m.from_square == from_sq and m.to_square == to_sq
                        for m in board.legal_moves
                    )
                    assert found, f"Policy nonzero at ({from_sq},{to_sq}) but no legal move"
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py -v`
Expected: ImportError (or skip if no stockfish).

**Step 3: Implement**

`src/denoisr/data/stockfish_oracle.py`:

```python
import chess
import chess.engine
import torch

from denoisr.types import PolicyTarget, ValueTarget


class StockfishOracle:
    def __init__(self, path: str, depth: int = 12) -> None:
        self._engine = chess.engine.SimpleEngine.popen_uci(path)
        self._depth = depth

    def evaluate(
        self, board: chess.Board
    ) -> tuple[PolicyTarget, ValueTarget, float]:
        policy = self._get_policy(board)
        value, cp = self._get_value(board)
        return policy, value, cp

    def _get_policy(self, board: chess.Board) -> PolicyTarget:
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            return PolicyTarget(torch.zeros(64, 64, dtype=torch.float32))

        scores: list[float] = []
        for move in legal_moves:
            board.push(move)
            info = self._engine.analyse(
                board, chess.engine.Limit(depth=max(1, self._depth - 2))
            )
            score = info["score"].white()
            cp_val = score.score(mate_score=10000)
            if cp_val is None:
                cp_val = 0
            if board.turn == chess.WHITE:
                cp_val = -cp_val
            scores.append(float(cp_val))
            board.pop()

        # Softmax over centipawn scores to get distribution
        t = torch.tensor(scores, dtype=torch.float32)
        probs = torch.softmax(t / 100.0, dim=0)

        data = torch.zeros(64, 64, dtype=torch.float32)
        for move, prob in zip(legal_moves, probs):
            data[move.from_square, move.to_square] = prob.item()

        return PolicyTarget(data)

    def _get_value(self, board: chess.Board) -> tuple[ValueTarget, float]:
        info = self._engine.analyse(
            board, chess.engine.Limit(depth=self._depth)
        )
        score = info["score"].white()
        cp_val = score.score(mate_score=10000)
        if cp_val is None:
            cp_val = 0

        wdl = info.get("wdl")
        if wdl is not None:
            w, d, l = wdl.white().wdl().tuple()
            total = w + d + l
            value = ValueTarget(
                win=w / total, draw=d / total, loss=l / total
            )
        else:
            # Approximate WDL from centipawns using sigmoid
            win_prob = 1.0 / (1.0 + 10.0 ** (-float(cp_val) / 400.0))
            value = ValueTarget(
                win=win_prob, draw=0.0, loss=1.0 - win_prob
            )

        return value, float(cp_val)

    def close(self) -> None:
        self._engine.quit()

    def __enter__(self) -> "StockfishOracle":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data/test_stockfish_oracle.py -v`
Expected: All pass (or skipped if no stockfish).

**Step 5: Commit**

```bash
git add src/denoisr/data/stockfish_oracle.py tests/test_data/test_stockfish_oracle.py
git commit -m "feat: add Stockfish oracle for policy/value targets (Layer 2d)"
```

---

### Task 7: Chess Dataset + Example Generation

**Scaling notes:** For datasets under 100M positions, use NumPy memory-mapped files. For larger collections, use HDF5. Use PyTorch's `DataLoader` with `num_workers > 0` and `pin_memory=True`. Typical batch sizes for chess position networks range from **4,096 to 16,384**.

**Files:**

- Create: `src/denoisr/data/dataset.py`
- Test: `tests/test_data/test_dataset.py`

**Step 1: Write failing tests**

`tests/test_data/test_dataset.py`:

```python
import chess
import pytest
import torch

from denoisr.data.board_encoder import SimpleBoardEncoder
from denoisr.data.dataset import ChessDataset, generate_examples_from_game
from denoisr.types import (
    Action,
    BoardTensor,
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


def _make_example(result: float = 1.0) -> TrainingExample:
    return TrainingExample(
        board=BoardTensor(torch.zeros(12, 8, 8)),
        policy=PolicyTarget(torch.zeros(64, 64)),
        value=ValueTarget(
            win=max(result, 0.0),
            draw=1.0 - abs(result),
            loss=max(-result, 0.0),
        ),
    )


class TestChessDataset:
    def test_length(self) -> None:
        examples = [_make_example() for _ in range(10)]
        ds = ChessDataset(examples)
        assert len(ds) == 10

    def test_getitem_shapes(self) -> None:
        ds = ChessDataset([_make_example()])
        board, policy, value = ds[0]
        assert board.shape == (12, 8, 8)
        assert policy.shape == (64, 64)
        assert value.shape == (3,)

    def test_value_tensor_order(self) -> None:
        ex = _make_example(result=1.0)
        ds = ChessDataset([ex])
        _, _, value = ds[0]
        assert value[0].item() == ex.value.win
        assert value[1].item() == ex.value.draw
        assert value[2].item() == ex.value.loss


class TestGenerateExamples:
    def test_scholars_mate(self) -> None:
        record = GameRecord(
            actions=(
                Action(12, 28),   # e2e4
                Action(52, 36),   # e7e5
                Action(3, 39),    # d1h5 (actually Qh5 = d1->h5? let me use proper squares)
            ),
            result=1.0,
        )
        # Use actual Scholar's Mate moves
        board = chess.Board()
        moves = ["e2e4", "e7e5", "d1h5", "b8c6", "f1c4", "g8f6", "h5f7"]
        record = GameRecord(
            actions=tuple(
                Action(
                    chess.Move.from_uci(m).from_square,
                    chess.Move.from_uci(m).to_square,
                    chess.Move.from_uci(m).promotion,
                )
                for m in moves
            ),
            result=1.0,
        )
        encoder = SimpleBoardEncoder()
        examples = list(generate_examples_from_game(record, encoder))
        assert len(examples) == 7  # one per position before each move
        for ex in examples:
            assert ex.board.data.shape == (12, 8, 8)
            assert ex.value.win + ex.value.draw + ex.value.loss == pytest.approx(1.0)

    def test_policy_target_has_played_move(self) -> None:
        board = chess.Board()
        move = chess.Move.from_uci("e2e4")
        record = GameRecord(
            actions=(Action(move.from_square, move.to_square, move.promotion),),
            result=0.0,
        )
        encoder = SimpleBoardEncoder()
        examples = list(generate_examples_from_game(record, encoder))
        assert len(examples) == 1
        assert examples[0].policy.data[move.from_square, move.to_square].item() == 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_dataset.py -v`
Expected: ImportError.

**Step 3: Implement**

`src/denoisr/data/dataset.py`:

```python
from collections.abc import Iterator

import chess
import torch
from torch.utils.data import Dataset

from denoisr.data.protocols import BoardEncoder
from denoisr.types import (
    Action,
    GameRecord,
    PolicyTarget,
    TrainingExample,
    ValueTarget,
)


class ChessDataset(Dataset[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]):
    def __init__(self, examples: list[TrainingExample]) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ex = self._examples[idx]
        value = torch.tensor(
            [ex.value.win, ex.value.draw, ex.value.loss],
            dtype=torch.float32,
        )
        return ex.board.data, ex.policy.data, value


def generate_examples_from_game(
    record: GameRecord, encoder: BoardEncoder
) -> Iterator[TrainingExample]:
    """Generate training examples from a game record using the played move as policy target."""
    board = chess.Board()
    for action in record.actions:
        board_tensor = encoder.encode(board)

        policy_data = torch.zeros(64, 64, dtype=torch.float32)
        policy_data[action.from_square, action.to_square] = 1.0
        policy = PolicyTarget(policy_data)

        if record.result == 1.0:
            value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        elif record.result == -1.0:
            value = ValueTarget(win=0.0, draw=0.0, loss=1.0)
        else:
            value = ValueTarget(win=0.0, draw=1.0, loss=0.0)

        yield TrainingExample(board=board_tensor, policy=policy, value=value)

        move = chess.Move(
            action.from_square, action.to_square, action.promotion
        )
        board.push(move)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data/test_dataset.py -v`
Expected: All pass.

**Step 5: Run ALL tests for Tiers 1–3**

Run: `uv run pytest tests/ -v`
Expected: All pass (stockfish tests skipped if binary not found).

**Step 6: Commit**

```bash
git add src/denoisr/data/dataset.py tests/test_data/test_dataset.py
git commit -m "feat: add chess dataset and example generation (Layer 2e)"
```

---

### Task 8: Extended Board Encoder (AlphaVile FX)

**Spec reference:** AlphaVile extended features (+180 Elo), input representation section. Shaw relative PE requires richer input but is handled in the NN plan.

**Rationale:** The 12-plane `SimpleBoardEncoder` only captures piece placement. The spec describes ~112 input planes including:

- 12 planes: piece placement (existing)
- 12 × 7 = 84 planes: past 7 positions (history stack)
- 4 planes: castling rights (K, Q, k, q as full-board binary)
- 1 plane: en passant target square
- 1 plane: rule-50 counter (normalized to [0, 1])
- 1 plane: repetition count (0/1/2 normalized)
- 1 plane: side to move
- 2 planes: material count per side (normalized)
- 2 planes: pieces giving check (per side)
- 2 planes: opposite-colored bishops flag (broadcast)

Total: ~112 planes (exact count determined during implementation).

**Files:**

- Create: `src/denoisr/data/extended_board_encoder.py`
- Test: `tests/test_data/test_extended_board_encoder.py`

**Step 1: Write failing tests**

`tests/test_data/test_extended_board_encoder.py`:

```python
import chess
import pytest
import torch
from hypothesis import given, settings

from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.types import BoardTensor

from tests.conftest import random_boards


class TestExtendedBoardEncoder:
    @pytest.fixture
    def encoder(self) -> ExtendedBoardEncoder:
        return ExtendedBoardEncoder()

    def test_num_planes_exceeds_simple(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        assert encoder.num_planes > 12

    def test_starting_position_shape(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        bt = encoder.encode(chess.Board())
        assert bt.data.shape == (encoder.num_planes, 8, 8)

    def test_piece_planes_match_simple_encoder(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        bt = encoder.encode(board)
        # First 12 planes should be piece placement
        assert bt.data[:12].sum().item() == 32

    def test_castling_rights_change(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board_with = chess.Board()
        board_without = chess.Board()
        board_without.set_castling_fen("-")
        bt_with = encoder.encode(board_with)
        bt_without = encoder.encode(board_without)
        assert not torch.equal(bt_with.data, bt_without.data)

    def test_en_passant_encoded(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        board.push_uci("e2e4")  # creates en passant target
        bt = encoder.encode(board)
        # Should differ from starting position encoding
        bt_start = encoder.encode(chess.Board())
        assert not torch.equal(bt.data, bt_start.data)

    def test_side_to_move_differs(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board_white = chess.Board()
        board_black = chess.Board()
        board_black.push_uci("e2e4")
        bt_w = encoder.encode(board_white)
        bt_b = encoder.encode(board_black)
        assert not torch.equal(bt_w.data, bt_b.data)

    def test_deterministic(
        self, encoder: ExtendedBoardEncoder
    ) -> None:
        board = chess.Board()
        bt1 = encoder.encode(board)
        bt2 = encoder.encode(board)
        assert torch.equal(bt1.data, bt2.data)

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_no_nan_or_inf(
        self, encoder: ExtendedBoardEncoder, board: chess.Board
    ) -> None:
        bt = encoder.encode(board)
        assert not torch.isnan(bt.data).any()
        assert not torch.isinf(bt.data).any()

    @given(board=random_boards())
    @settings(max_examples=30)
    def test_values_bounded(
        self, encoder: ExtendedBoardEncoder, board: chess.Board
    ) -> None:
        bt = encoder.encode(board)
        assert bt.data.min() >= 0.0
        assert bt.data.max() <= 1.0
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_extended_board_encoder.py -v`
Expected: ImportError.

**Step 3: Implement**

`src/denoisr/data/extended_board_encoder.py`:

```python
import chess
import torch

from denoisr.types import BoardTensor

_PLANE_INDEX = {
    (pt, color): (pt - 1) + (0 if color == chess.WHITE else 6)
    for pt in chess.PIECE_TYPES
    for color in chess.COLORS
}

_HISTORY_DEPTH = 7
_PIECE_PLANES = 12
_HISTORY_PLANES = _PIECE_PLANES * _HISTORY_DEPTH  # 84
_META_PLANES = 14  # castling(4) + ep(1) + rule50(1) + repetition(1) + stm(1) + material(2) + check(2) + opp_bishops(2)
_TOTAL_PLANES = _PIECE_PLANES + _HISTORY_PLANES + _META_PLANES  # 112


def _encode_pieces(board: chess.Board, planes: torch.Tensor, offset: int) -> None:
    for sq, piece in board.piece_map().items():
        plane = offset + _PLANE_INDEX[(piece.piece_type, piece.color)]
        rank = chess.square_rank(sq)
        file = chess.square_file(sq)
        planes[plane, rank, file] = 1.0


class ExtendedBoardEncoder:
    """Extended board encoder with AlphaVile features (~112 planes).

    Encodes: current + 7 history positions, castling rights, en passant,
    rule-50, repetition count, side to move, material counts,
    pieces giving check, and opposite-colored bishops.
    """

    @property
    def num_planes(self) -> int:
        return _TOTAL_PLANES

    def encode(self, board: chess.Board) -> BoardTensor:
        data = torch.zeros(_TOTAL_PLANES, 8, 8, dtype=torch.float32)

        # Current position piece planes (0..11)
        _encode_pieces(board, data, 0)

        # History planes (12..95): up to 7 past positions
        history_board = board.copy()
        for h in range(_HISTORY_DEPTH):
            if not history_board.move_stack:
                break
            history_board.pop()
            offset = _PIECE_PLANES + h * _PIECE_PLANES
            _encode_pieces(history_board, data, offset)

        meta_start = _PIECE_PLANES + _HISTORY_PLANES

        # Castling rights (4 planes, broadcast)
        for i, right in enumerate([
            chess.BB_H1,  # white kingside
            chess.BB_A1,  # white queenside
            chess.BB_H8,  # black kingside
            chess.BB_A8,  # black queenside
        ]):
            has_right = bool(board.castling_rights & right)
            if has_right:
                data[meta_start + i] = 1.0

        # En passant (1 plane)
        if board.ep_square is not None:
            rank = chess.square_rank(board.ep_square)
            file = chess.square_file(board.ep_square)
            data[meta_start + 4, rank, file] = 1.0

        # Rule-50 counter (1 plane, normalized)
        data[meta_start + 5] = min(board.halfmove_clock / 100.0, 1.0)

        # Repetition count (1 plane, normalized: 0, 0.5, 1.0)
        if board.is_repetition(2):
            data[meta_start + 6] = 1.0
        elif board.is_repetition(1):
            data[meta_start + 6] = 0.5

        # Side to move (1 plane)
        if board.turn == chess.WHITE:
            data[meta_start + 7] = 1.0

        # Material counts (2 planes, normalized by max possible)
        for ci, color in enumerate(chess.COLORS):
            material = sum(
                len(board.pieces(pt, color)) * v
                for pt, v in zip(chess.PIECE_TYPES, [1, 3, 3, 5, 9, 0])
            )
            data[meta_start + 8 + ci] = min(material / 39.0, 1.0)

        # Pieces giving check (2 planes)
        for ci, color in enumerate(chess.COLORS):
            king_sq = board.king(not color)
            if king_sq is not None:
                attackers = board.attackers(color, king_sq)
                for sq in attackers:
                    rank = chess.square_rank(sq)
                    file = chess.square_file(sq)
                    data[meta_start + 10 + ci, rank, file] = 1.0

        # Opposite-colored bishops (2 planes, broadcast)
        white_bishops = board.pieces(chess.BISHOP, chess.WHITE)
        black_bishops = board.pieces(chess.BISHOP, chess.BLACK)
        if white_bishops and black_bishops:
            w_light = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 0
                for sq in white_bishops
            )
            w_dark = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 1
                for sq in white_bishops
            )
            b_light = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 0
                for sq in black_bishops
            )
            b_dark = any(
                (chess.square_rank(sq) + chess.square_file(sq)) % 2 == 1
                for sq in black_bishops
            )
            if (w_light and b_dark and not w_dark and not b_light) or \
               (w_dark and b_light and not w_light and not b_dark):
                data[meta_start + 12] = 1.0
                data[meta_start + 13] = 1.0

        return BoardTensor(data)
```

**Step 4: Run tests**

Run: `uv run pytest tests/test_data/test_extended_board_encoder.py -v`
Expected: All pass.

**Step 5: Commit**

```bash
git add src/denoisr/data/extended_board_encoder.py tests/test_data/test_extended_board_encoder.py
git commit -m "feat: add extended board encoder with AlphaVile features (Layer 2f)"
```

---

## Tier 1–3 Gate Check

After all 9 tasks, run the full test suite:

```bash
uv run pytest tests/ -v --tb=short
```

**Expected:** All tests pass (stockfish tests skip gracefully). This validates the complete foundation: types, game interface, board encoder, action encoder, PGN streaming, Stockfish oracle, and dataset construction.

**Next plan:** `2026-02-19-neural-networks-implementation.md` covering Tier 4 (encoder, smolgen, policy head, value head) and Tier 5 (policy backbone, world model, consistency, diffusion).
