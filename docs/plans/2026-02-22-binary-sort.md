# Binary Game Format + Data Path Unification Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace PGN-text bucket files with a custom binary `.games` format, unify `pgn_path`/`sorted_dir`/`elo_dir` into a single `data_dir`, and add batched disk writes with 16 GB per-bucket buffers.

**Architecture:** New `game_format.py` module handles binary serialization/deserialization. `sort_pgn.py` writes `.games` files via `GameBatchWriter` instead of reconstructing PGN text. `generate_data.py` reads `.games` files directly, eliminating the dual `_stream_single_pgn`/`_stream_elo_buckets` code paths. `DataConfig` drops `pgn_path`/`sorted_dir` in favor of `data_dir`.

**Tech Stack:** Python `struct` module for binary I/O, `bytearray` for batch buffers, existing `SimplePGNStreamer` for reading source PGN.

**Design doc:** `docs/plans/2026-02-22-binary-sort-design.md`

---

### Task 1: Binary Game Format — Serialization Primitives

**Files:**
- Create: `src/denoisr/data/game_format.py`
- Test: `tests/test_data/test_game_format.py`

**Step 1: Write the failing tests**

```python
# tests/test_data/test_game_format.py
"""Tests for the binary .games format."""

import struct
from pathlib import Path

from denoisr.data.game_format import (
    FILE_MAGIC,
    FORMAT_VERSION,
    GameBatchWriter,
    count_positions,
    read_game_records,
    serialize_record,
    write_file_header,
)
from denoisr.types import Action, GameRecord


class TestSerializeRecord:
    def test_round_trip_basic_game(self) -> None:
        record = GameRecord(
            actions=(Action(12, 28), Action(52, 36), Action(1, 18)),
            result=1.0,
            white_elo=1500,
            black_elo=1400,
            eco_code="B01",
        )
        data = serialize_record(record)
        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_round_trip_no_elo(self) -> None:
        record = GameRecord(
            actions=(Action(12, 28),),
            result=0.0,
            white_elo=None,
            black_elo=None,
        )
        data = serialize_record(record)
        # -1 sentinel for missing Elo
        result, w_elo, b_elo = struct.unpack_from("<fhh", data)
        assert w_elo == -1
        assert b_elo == -1

    def test_round_trip_no_eco(self) -> None:
        record = GameRecord(
            actions=(Action(12, 28),),
            result=-1.0,
        )
        data = serialize_record(record)
        # eco_len should be 0
        eco_len = struct.unpack_from("<B", data, 8)[0]
        assert eco_len == 0

    def test_round_trip_with_promotion(self) -> None:
        record = GameRecord(
            actions=(Action(52, 60, promotion=5),),  # queen promotion
            result=1.0,
        )
        data = serialize_record(record)
        assert len(data) > 0

    def test_round_trip_empty_game(self) -> None:
        record = GameRecord(actions=(), result=0.0)
        data = serialize_record(record)
        # num_moves should be 0
        num_moves = struct.unpack_from("<H", data, 9)[0]
        assert num_moves == 0


class TestFileIO:
    def test_write_and_read_single_record(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        record = GameRecord(
            actions=(Action(12, 28), Action(52, 36)),
            result=1.0,
            white_elo=1500,
            black_elo=1400,
            eco_code="C00",
        )

        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert len(records) == 1
        r = records[0]
        assert r.result == 1.0
        assert r.white_elo == 1500
        assert r.black_elo == 1400
        assert r.eco_code == "C00"
        assert len(r.actions) == 2
        assert r.actions[0] == Action(12, 28)
        assert r.actions[1] == Action(52, 36)

    def test_write_and_read_multiple_records(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        records_in = [
            GameRecord(actions=(Action(i, i + 1),), result=1.0, white_elo=1000 + i)
            for i in range(10)
        ]

        with open(path, "wb") as fh:
            write_file_header(fh)
            for rec in records_in:
                fh.write(serialize_record(rec))

        records_out = list(read_game_records(path))
        assert len(records_out) == 10
        for i, r in enumerate(records_out):
            assert r.actions[0].from_square == i
            assert r.white_elo == 1000 + i

    def test_missing_elo_round_trips_as_none(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        record = GameRecord(actions=(Action(0, 1),), result=0.0)

        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert records[0].white_elo is None
        assert records[0].black_elo is None

    def test_promotion_round_trips(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        record = GameRecord(
            actions=(Action(52, 60, promotion=5), Action(12, 28)),
            result=1.0,
        )

        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert records[0].actions[0].promotion == 5
        assert records[0].actions[1].promotion is None

    def test_file_magic_validation(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.games"
        path.write_bytes(b"BAAD\x01\x00\x00\x00")

        import pytest
        with pytest.raises(ValueError, match="magic"):
            list(read_game_records(path))


class TestCountPositions:
    def test_counts_total_moves(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        records = [
            GameRecord(actions=(Action(0, 1), Action(2, 3), Action(4, 5)), result=1.0),
            GameRecord(actions=(Action(6, 7),), result=0.0),
        ]

        with open(path, "wb") as fh:
            write_file_header(fh)
            for rec in records:
                fh.write(serialize_record(rec))

        assert count_positions(path) == 4  # 3 + 1


class TestGameBatchWriter:
    def test_flush_on_close(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        records = [
            GameRecord(actions=(Action(i, i + 1),), result=1.0)
            for i in range(5)
        ]

        with GameBatchWriter(path) as writer:
            for rec in records:
                writer.write(rec)
            # No explicit flush — close should handle it

        read_back = list(read_game_records(path))
        assert len(read_back) == 5

    def test_flush_when_buffer_exceeds_max(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        # Use tiny max_buffer_bytes to force mid-write flush
        with GameBatchWriter(path, max_buffer_bytes=50) as writer:
            for i in range(20):
                writer.write(GameRecord(actions=(Action(i % 63, (i + 1) % 64),), result=1.0))

        read_back = list(read_game_records(path))
        assert len(read_back) == 20

    def test_empty_writer_produces_valid_file(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        with GameBatchWriter(path) as writer:
            pass  # write nothing

        records = list(read_game_records(path))
        assert records == []

    def test_writer_count_tracks_records(self, tmp_path: Path) -> None:
        path = tmp_path / "test.games"
        with GameBatchWriter(path) as writer:
            for i in range(7):
                writer.write(GameRecord(actions=(Action(0, 1),), result=0.0))
            assert writer.count == 7
```

**Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_data/test_game_format.py -v`
Expected: FAIL — `ImportError: cannot import name 'FILE_MAGIC' from 'denoisr.data.game_format'`

**Step 3: Write the implementation**

```python
# src/denoisr/data/game_format.py
"""Binary .games format for Elo-sorted game records.

File layout:
    [file header: 8 bytes]
        magic:   b"DNSR" (4 bytes)
        version: uint8 (1 byte, currently 1)
        _pad:    3 bytes

    [record]*
        [header: 14 bytes]
            result:    float32  (4 bytes)
            white_elo: int16    (2 bytes, -1 = missing)
            black_elo: int16    (2 bytes, -1 = missing)
            eco_len:   uint8    (1 byte, 0 = no ECO)
            num_moves: uint16   (2 bytes)
            _pad:      1 byte
        [eco_code: eco_len bytes, UTF-8]
        [moves: num_moves * 3 bytes]
            from_sq:   uint8
            to_sq:     uint8
            promotion: uint8 (0xFF = none)
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO

from denoisr.types import Action, GameRecord

FILE_MAGIC = b"DNSR"
FORMAT_VERSION = 1

# struct formats
_FILE_HEADER = struct.Struct("<4sB3x")       # magic(4) + version(1) + pad(3) = 8
_RECORD_HEADER = struct.Struct("<fhhBHx")    # result(4) + w_elo(2) + b_elo(2) + eco_len(1) + num_moves(2) + pad(1) = 14
_MOVE = struct.Struct("<BBB")                # from(1) + to(1) + promo(1) = 3

_NO_PROMOTION = 0xFF


def write_file_header(fh: BinaryIO) -> None:
    """Write the 8-byte file header."""
    fh.write(_FILE_HEADER.pack(FILE_MAGIC, FORMAT_VERSION))


def _read_file_header(fh: BinaryIO) -> None:
    """Read and validate the file header."""
    data = fh.read(_FILE_HEADER.size)
    if len(data) < _FILE_HEADER.size:
        raise ValueError("File too short for header")
    magic, version = _FILE_HEADER.unpack(data)
    if magic != FILE_MAGIC:
        raise ValueError(f"Invalid magic: expected {FILE_MAGIC!r}, got {magic!r}")
    if version != FORMAT_VERSION:
        raise ValueError(f"Unsupported version: {version}")


def serialize_record(record: GameRecord) -> bytes:
    """Serialize a GameRecord to bytes."""
    eco_bytes = record.eco_code.encode("utf-8") if record.eco_code else b""
    header = _RECORD_HEADER.pack(
        record.result,
        record.white_elo if record.white_elo is not None else -1,
        record.black_elo if record.black_elo is not None else -1,
        len(eco_bytes),
        len(record.actions),
    )
    moves = b"".join(
        _MOVE.pack(
            a.from_square,
            a.to_square,
            a.promotion if a.promotion is not None else _NO_PROMOTION,
        )
        for a in record.actions
    )
    return header + eco_bytes + moves


def _read_record(fh: BinaryIO) -> GameRecord | None:
    """Read one record from the file, or None at EOF."""
    header_data = fh.read(_RECORD_HEADER.size)
    if len(header_data) == 0:
        return None
    if len(header_data) < _RECORD_HEADER.size:
        raise ValueError("Truncated record header")
    result, w_elo, b_elo, eco_len, num_moves = _RECORD_HEADER.unpack(header_data)

    eco_code: str | None = None
    if eco_len > 0:
        eco_bytes = fh.read(eco_len)
        eco_code = eco_bytes.decode("utf-8")

    actions: list[Action] = []
    for _ in range(num_moves):
        move_data = fh.read(_MOVE.size)
        from_sq, to_sq, promo = _MOVE.unpack(move_data)
        promotion = None if promo == _NO_PROMOTION else promo
        actions.append(Action(from_sq, to_sq, promotion))

    return GameRecord(
        actions=tuple(actions),
        result=result,
        white_elo=w_elo if w_elo != -1 else None,
        black_elo=b_elo if b_elo != -1 else None,
        eco_code=eco_code,
    )


def read_game_records(path: Path) -> Iterator[GameRecord]:
    """Iterate over all GameRecords in a .games file."""
    with open(path, "rb") as fh:
        _read_file_header(fh)
        while True:
            record = _read_record(fh)
            if record is None:
                break
            yield record


def count_positions(path: Path) -> int:
    """Count total positions (moves) across all records without full deserialization.

    Reads only record headers, skipping ECO and move data.
    """
    total = 0
    with open(path, "rb") as fh:
        _read_file_header(fh)
        while True:
            header_data = fh.read(_RECORD_HEADER.size)
            if len(header_data) == 0:
                break
            if len(header_data) < _RECORD_HEADER.size:
                raise ValueError("Truncated record header")
            _, _, _, eco_len, num_moves = _RECORD_HEADER.unpack(header_data)
            total += num_moves
            # Skip eco + moves
            fh.seek(eco_len + num_moves * _MOVE.size, 1)
    return total


@dataclass
class GameBatchWriter:
    """Buffered writer for .games files.

    Accumulates serialized records in memory and flushes to disk when the
    buffer exceeds ``max_buffer_bytes`` or on close.
    """

    path: Path
    max_buffer_bytes: int = 16 * 1024**3  # 16 GB
    _buffer: bytearray = field(default_factory=bytearray, init=False, repr=False)
    _fh: BinaryIO | None = field(default=None, init=False, repr=False)
    _count: int = field(default=0, init=False)

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self) -> GameBatchWriter:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "wb")  # noqa: SIM115
        write_file_header(self._fh)
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def write(self, record: GameRecord) -> None:
        self._buffer.extend(serialize_record(record))
        self._count += 1
        if len(self._buffer) >= self.max_buffer_bytes:
            self.flush()

    def flush(self) -> None:
        if self._buffer and self._fh is not None:
            self._fh.write(bytes(self._buffer))
            self._buffer.clear()

    def close(self) -> None:
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_data/test_game_format.py -v`
Expected: All pass

**Step 5: Commit**

```
git add src/denoisr/data/game_format.py tests/test_data/test_game_format.py
git commit -m "feat: add binary .games format with batch writer"
```

---

### Task 2: Rewrite sort_pgn to Emit Binary .games Files

**Files:**
- Modify: `src/denoisr/scripts/sort_pgn.py`
- Modify: `tests/test_data/test_sort_pgn.py` (if exists, else create)

**Step 1: Write the failing test**

Check if sort_pgn tests exist. If not, create them. The key behavioral test: given a PGN file with games at different Elos, sort_pgn should produce `.games` files per bucket with correct records.

```python
# tests/test_scripts/test_sort_pgn.py
"""Tests for PGN-to-binary sort."""

from pathlib import Path

import chess
import chess.pgn

from denoisr.data.game_format import read_game_records


def _write_test_pgn(path: Path, games: list[dict]) -> None:
    """Write a simple PGN file with the given game metadata."""
    with open(path, "w") as f:
        for g in games:
            game = chess.pgn.Game()
            game.headers["WhiteElo"] = str(g["white_elo"])
            game.headers["BlackElo"] = str(g["black_elo"])
            game.headers["Result"] = g.get("result", "1-0")
            if "eco" in g:
                game.headers["ECO"] = g["eco"]
            # Add a few moves
            node = game
            board = chess.Board()
            for uci in g.get("moves", ["e2e4", "e7e5"]):
                move = chess.Move.from_uci(uci)
                node = node.add_variation(move)
                board.push(move)
            f.write(str(game) + "\n\n")


def test_sort_creates_games_files(tmp_path: Path) -> None:
    """sort_pgn creates .games files for each Elo bucket."""
    from denoisr.scripts.sort_pgn import sort_pgn_to_games

    pgn = tmp_path / "input.pgn"
    out = tmp_path / "sorted"
    _write_test_pgn(pgn, [
        {"white_elo": 900, "black_elo": 850, "moves": ["e2e4", "e7e5"]},
        {"white_elo": 1500, "black_elo": 1400, "moves": ["d2d4", "d7d5"]},
        {"white_elo": 2500, "black_elo": 2400, "eco": "B01", "moves": ["e2e4", "d7d5"]},
    ])

    ranges = [(0, 1200), (1200, 2000), (2000, None)]
    stats = sort_pgn_to_games(pgn, out, ranges)

    assert (out / "0-1200.games").exists()
    assert (out / "1200-2000.games").exists()
    assert (out / "2000+.games").exists()

    # Verify contents
    low_games = list(read_game_records(out / "0-1200.games"))
    assert len(low_games) == 1
    assert low_games[0].white_elo == 900

    mid_games = list(read_game_records(out / "1200-2000.games"))
    assert len(mid_games) == 1
    assert mid_games[0].white_elo == 1500

    high_games = list(read_game_records(out / "2000+.games"))
    assert len(high_games) == 1
    assert high_games[0].eco_code == "B01"

    assert stats["0-1200"] == 1
    assert stats["1200-2000"] == 1
    assert stats["2000+"] == 1


def test_sort_skips_games_without_elo(tmp_path: Path) -> None:
    """Games without Elo headers are skipped."""
    from denoisr.scripts.sort_pgn import sort_pgn_to_games

    pgn = tmp_path / "input.pgn"
    # Write a game with no Elo using raw PGN text
    pgn.write_text(
        '[Result "1-0"]\n\n1. e4 e5 *\n\n'
        '[WhiteElo "1500"]\n[BlackElo "1400"]\n[Result "1-0"]\n\n1. d4 d5 *\n\n'
    )

    out = tmp_path / "sorted"
    ranges = [(0, 2000), (2000, None)]
    stats = sort_pgn_to_games(pgn, out, ranges)

    total = sum(stats.values())
    assert total == 1  # only the game with Elo
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_scripts/test_sort_pgn.py -v`
Expected: FAIL — `ImportError: cannot import name 'sort_pgn_to_games'`

**Step 3: Rewrite sort_pgn.py**

Replace the PGN-text output logic with binary `.games` output. Extract the core logic
into a `sort_pgn_to_games()` function (testable), keep `main()` as the CLI wrapper.

The key changes:
- Replace zstd-compressed PGN text writers with `GameBatchWriter`
- Remove `chess.pgn.Game()` reconstruction (the heaviest part)
- `GameRecord` from the streamer goes directly to binary serialization
- `_bucket_name()` stays the same, file extension changes from `.pgn.zst` to `.games`

```python
# src/denoisr/scripts/sort_pgn.py
"""Sort PGN games into Elo-stratified binary .games files.

Reads a .pgn or .pgn.zst file and writes separate .games files
per Elo range under the output directory.
"""

import argparse
import logging
from pathlib import Path

from denoisr.data.game_format import GameBatchWriter
from denoisr.data.pgn_streamer import SimplePGNStreamer

log = logging.getLogger(__name__)


def _parse_ranges(raw: str) -> list[tuple[int, int | None]]:
    """Parse '0-1200,1200-1600,2400+' into [(0,1200),(1200,1600),(2400,None)]."""
    ranges: list[tuple[int, int | None]] = []
    for part in raw.split(","):
        part = part.strip()
        if part.endswith("+"):
            ranges.append((int(part[:-1]), None))
        else:
            lo, hi = part.split("-")
            ranges.append((int(lo), int(hi)))
    return ranges


def _bucket_name(lo: int, hi: int | None) -> str:
    if hi is None:
        return f"{lo}+"
    return f"{lo}-{hi}"


def _min_elo(white_elo: int | None, black_elo: int | None) -> int | None:
    if white_elo is not None and black_elo is not None:
        return min(white_elo, black_elo)
    return white_elo or black_elo


def sort_pgn_to_games(
    pgn_path: Path,
    output_dir: Path,
    ranges: list[tuple[int, int | None]],
    max_buffer_bytes: int = 16 * 1024**3,
) -> dict[str, int]:
    """Sort PGN games into binary .games files by Elo range.

    Returns a dict mapping bucket names to game counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[str, GameBatchWriter] = {}
    counts: dict[str, int] = {}
    skipped = 0

    # Open writers for each bucket
    for lo, hi in ranges:
        name = _bucket_name(lo, hi)
        path = output_dir / f"{name}.games"
        writers[name] = GameBatchWriter(path, max_buffer_bytes=max_buffer_bytes)
        writers[name].__enter__()
        counts[name] = 0

    streamer = SimplePGNStreamer()
    try:
        for record in streamer.stream(pgn_path):
            elo = _min_elo(record.white_elo, record.black_elo)
            if elo is None:
                skipped += 1
                continue

            bucket_name: str | None = None
            for lo, hi in ranges:
                if hi is None:
                    if elo >= lo:
                        bucket_name = _bucket_name(lo, hi)
                        break
                elif lo <= elo < hi:
                    bucket_name = _bucket_name(lo, hi)
                    break

            if bucket_name is None:
                skipped += 1
                continue

            writers[bucket_name].write(record)
            counts[bucket_name] += 1
    finally:
        for writer in writers.values():
            writer.__exit__(None, None, None)

    log.info("Sorted games by Elo (skipped %d without Elo):", skipped)
    for name, count in counts.items():
        log.info("  %s: %d games", name, count)

    return counts


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(
        description="Sort PGN games by Elo into binary .games files"
    )
    parser.add_argument("--input", required=True, help="Path to .pgn or .pgn.zst")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument(
        "--ranges",
        default="0-1200,1200-1600,1600-2000,2000-2400,2400+",
        help="Comma-separated Elo ranges (default: 0-1200,1200-1600,1600-2000,2000-2400,2400+)",
    )
    args = parser.parse_args()
    ranges = _parse_ranges(args.ranges)
    sort_pgn_to_games(Path(args.input), Path(args.output), ranges)


if __name__ == "__main__":
    main()
```

**Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_scripts/test_sort_pgn.py tests/test_data/test_game_format.py -v`
Expected: All pass

**Step 5: Commit**

```
git add src/denoisr/scripts/sort_pgn.py tests/test_scripts/test_sort_pgn.py
git commit -m "refactor: rewrite sort_pgn to emit binary .games files"
```

---

### Task 3: Unify DataConfig — Replace pgn_path/sorted_dir with data_dir

**Files:**
- Modify: `src/denoisr/pipeline/config.py`
- Modify: `pipeline.toml`
- Modify: `tests/test_pipeline/test_config.py`

**Step 1: Write the failing test**

```python
# Add to tests/test_pipeline/test_config.py

def test_data_dir_field_exists(tmp_path: Path) -> None:
    """DataConfig has data_dir field, not pgn_path or sorted_dir."""
    cfg_path = tmp_path / "pipeline.toml"
    cfg_path.write_text('[data]\ndata_dir = "my_data/"')
    cfg = load_config(cfg_path)
    assert cfg.data.data_dir == "my_data/"
    assert not hasattr(cfg.data, "pgn_path")
    assert not hasattr(cfg.data, "sorted_dir")
```

**Step 2: Run test to verify it fails**

Run: `uv run pytest tests/test_pipeline/test_config.py::test_data_dir_field_exists -v`
Expected: FAIL — `TypeError: DataConfig.__init__() got an unexpected keyword argument 'data_dir'`

**Step 3: Update DataConfig**

In `src/denoisr/pipeline/config.py`, replace `pgn_path` and `sorted_dir` with `data_dir`:

```python
@dataclass(frozen=True)
class DataConfig:
    pgn_url: str = (
        "https://database.lichess.org/standard/"
        "lichess_db_standard_rated_2025-01.pgn.zst"
    )
    data_dir: str = "data/"
    stockfish_path: str = ""
    stockfish_depth: int = 10
    examples_per_tier: int = 2_000_000
    tactical_fraction: float = 0.25
    workers: int = 0
    write_buffer_max_bytes: int = 16 * 1024**3
```

Update `pipeline.toml`:
```toml
[data]
pgn_url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
data_dir = "data/"
stockfish_path = ""
stockfish_depth = 10
examples_per_tier = 2_000_000
tactical_fraction = 0.25
workers = 0
```

**Step 4: Fix all existing tests that reference pgn_path or sorted_dir in DataConfig**

Update `_make_cfg()` in `test_steps.py`, `test_runner.py`, `test_integration.py`:

```python
data=DataConfig(
    pgn_url="https://example.com/test.pgn.zst",
    data_dir=str(tmp_path / "data"),
),
```

Remove any `pgn_path=...` or `sorted_dir=...` from test configs.

**Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/test_pipeline/ -v`
Expected: Config tests pass, steps/runner/integration tests fail (steps.py still references old fields — expected, fixed in Task 4)

**Step 6: Commit**

```
git add src/denoisr/pipeline/config.py pipeline.toml tests/test_pipeline/
git commit -m "refactor: replace pgn_path/sorted_dir with data_dir in DataConfig"
```

---

### Task 4: Update Pipeline Steps for data_dir + .games

**Files:**
- Modify: `src/denoisr/pipeline/steps.py`
- Modify: `src/denoisr/pipeline/runner.py` (if sort step skip logic changes)
- Test: `tests/test_pipeline/test_steps.py`

**Step 1: Update step_fetch_data**

Change from `cfg.data.pgn_path` to `Path(cfg.data.data_dir) / "raw.pgn.zst"`.

**Step 2: Update step_sort_pgn**

- Check for `*.games` files in `data_dir` instead of `*.pgn.zst` in `sorted_dir`
- Call `sort_pgn_to_games()` instead of `sort_main()` via sys.argv injection
- Pass `cfg.data.write_buffer_max_bytes` through

**Step 3: Update step_generate_tier_data**

- Remove `elo_dir` derivation from `sorted_dir`
- Pass `data_dir` directly — `generate_to_file` will read `*.games` files (Task 5)
- Remove `pgn_path=Path(cfg.data.pgn_path)` argument

**Step 4: Fix test assertions**

Update `test_fetch_data_skips_when_file_exists` to create file at `data_dir/raw.pgn.zst`.
Update `test_sort_pgn_skips_when_sorted_dir_has_files` to check for `*.games` files.
Update `test_generate_tier_data_calls_generate` to verify `data_dir` param instead of `pgn_path`.

**Step 5: Run tests**

Run: `uv run pytest tests/test_pipeline/test_steps.py -v`
Expected: Pass (some generate tests may still fail until Task 5 updates generate_data.py)

**Step 6: Commit**

```
git add src/denoisr/pipeline/steps.py tests/test_pipeline/test_steps.py
git commit -m "refactor: update pipeline steps for data_dir and .games format"
```

---

### Task 5: Refactor generate_data.py — Read .games, Remove Dual Paths

**Files:**
- Modify: `src/denoisr/scripts/generate_data.py`

**Step 1: Replace `_stream_single_pgn` and `_stream_elo_buckets` with `_stream_game_files`**

The new function reads `*.games` files from `data_dir` using `read_game_records()`,
replays moves to generate `_PositionMeta` objects, and supports tactical enrichment.

Key changes:
- Remove `pgn_path` parameter from `generate_to_file()`
- Remove `elo_dir` parameter from `generate_to_file()`
- Add `data_dir: Path` parameter to `generate_to_file()`
- Replace `_count_positions()` PGN variant with `count_positions()` from `game_format`
- Remove `_stream_positions()` dispatcher — `_stream_game_files()` is the only path
- Update `main()` CLI: replace `--pgn` and `--elo-dir` with `--data-dir`

**Step 2: Update `_count_positions` to use binary format**

```python
def _count_positions_from_games(data_dir: Path, max_positions: int) -> int:
    """Count positions across all .games files in data_dir."""
    from denoisr.data.game_format import count_positions
    total = 0
    for gf in sorted(data_dir.glob("*.games")):
        total += count_positions(gf)
        if total >= max_positions:
            return max_positions
    return min(total, max_positions)
```

**Step 3: Write `_stream_game_files`**

```python
def _stream_game_files(
    data_dir: Path,
    max_positions: int,
    tactical_fraction: float = 0.25,
    seed: int | None = None,
) -> Iterator[_PositionMeta]:
    """Stream positions from .games binary files with tactical enrichment."""
    from denoisr.data.game_format import read_game_records

    game_files = sorted(data_dir.glob("*.games"))
    if not game_files:
        return

    if seed is not None:
        random.seed(seed)

    positions_per_bucket = max_positions // len(game_files)
    count = 0
    tactical_count = 0
    game_id = 0

    for gf in game_files:
        bucket_count = 0
        for record in read_game_records(gf):
            if bucket_count >= positions_per_bucket and count < max_positions:
                break
            moves_so_far: _MoveSeq = []
            board = chess.Board()
            for action in record.actions:
                moves_so_far.append((action.from_square, action.to_square, action.promotion))
                board.push(chess.Move(action.from_square, action.to_square, action.promotion))
                if count >= max_positions:
                    break

                is_tactical = _is_tactical(board)
                current_ratio = tactical_count / max(count, 1)
                if is_tactical and current_ratio < tactical_fraction:
                    pass  # always include
                elif not is_tactical and current_ratio >= tactical_fraction:
                    pass  # always include
                elif random.random() > 0.5:
                    continue

                piece_count = bin(board.occupied).count("1")
                yield _PositionMeta(
                    moves=list(moves_so_far),
                    game_id=game_id,
                    eco_code=record.eco_code,
                    piece_count=piece_count,
                )
                count += 1
                bucket_count += 1
                if is_tactical:
                    tactical_count += 1

            game_id += 1
            if count >= max_positions:
                break
```

**Step 4: Update generate_to_file signature**

```python
def generate_to_file(
    data_dir: Path,           # was: pgn_path + elo_dir
    output_path: Path,
    stockfish_path: str,
    stockfish_depth: int,
    max_examples: int,
    num_workers: int,
    policy_temperature: float = 80.0,
    label_smoothing: float = 0.02,
    chunksize: int = 64,
    min_elo: int | None = None,   # kept for per-tier Elo filtering
    tactical_fraction: float = 0.0,
    seed: int | None = None,
) -> int:
```

**Step 5: Update CLI in main()**

Replace `--pgn` and `--elo-dir` with `--data-dir`:

```python
parser.add_argument("--data-dir", required=True, help="Directory with .games files")
```

Remove `--pgn` and `--elo-dir` arguments.

**Step 6: Run tests**

Run: `uv run pytest tests/ -x -q`
Expected: All pass (pipeline tests already mock `generate_to_file`)

**Step 7: Commit**

```
git add src/denoisr/scripts/generate_data.py
git commit -m "refactor: replace PGN streaming with .games binary reader in generate_data"
```

---

### Task 6: Update train_phase2.py for .games Format

**Files:**
- Modify: `src/denoisr/scripts/train_phase2.py`

**Step 1: Update extract_trajectories**

`extract_trajectories()` currently takes `pgn_path: Path` and uses `SimplePGNStreamer`.
Change it to take `data_dir: Path` and use `read_game_records()` from `game_format`:

```python
def extract_trajectories(
    data_dir: Path,             # was: pgn_path
    encoder: ExtendedBoardEncoder,
    seq_len: int,
    max_trajectories: int,
    min_elo: int | None = None,
) -> TrajectoryBatch:
    from denoisr.data.game_format import read_game_records

    # ... iterate over sorted(data_dir.glob("*.games"))
    # For each file, read_game_records(), filter by elo, extract trajectories
```

**Step 2: Update CLI**

Replace `--pgn` with `--data-dir` in the argument parser.

**Step 3: Run tests**

Run: `uv run pytest tests/test_scripts/ -v`
Expected: Pass

**Step 4: Commit**

```
git add src/denoisr/scripts/train_phase2.py
git commit -m "refactor: update train_phase2 to read .games files from data_dir"
```

---

### Task 7: Update README.md and pipeline.toml

**Files:**
- Modify: `README.md`
- Modify: `pipeline.toml`

**Step 1: Update pipeline.toml [data] section**

Remove `pgn_path` and `sorted_dir`, keep `data_dir`. Remove comments referencing
`sorted/` directory. Add comment for `write_buffer_max_bytes` if exposed.

**Step 2: Update README.md**

- Update the TOML example block in the pipeline section
- Replace `pgn_path` and `sorted_dir` references with `data_dir`
- Update `--pgn` and `--elo-dir` CLI docs to `--data-dir`
- Remove references to `denoisr-sort-pgn` producing `.pgn.zst` files — now produces `.games`
- Update the manual workflow examples (`denoisr-generate-data`, `denoisr-train-phase2`)

**Step 3: Run all tests to verify nothing broke**

Run: `uv run pytest tests/ -x -q`
Expected: All pass

**Step 4: Commit**

```
git add README.md pipeline.toml
git commit -m "docs: update README and pipeline.toml for data_dir + .games format"
```

---

### Task 8: Full Pipeline Test — End-to-End Verification

**Files:**
- Modify: `tests/test_pipeline/test_integration.py`

**Step 1: Update integration test configs**

Replace `pgn_path`/`sorted_dir` with `data_dir` in all `_make_cfg()` helpers.

**Step 2: Verify mock signatures match new step signatures**

Especially `_fake_generate` which may need updated kwargs (`data_dir` instead of `pgn_path`).

**Step 3: Run the full test suite**

Run: `uv run pytest tests/ -x -q`
Expected: All pass, 0 failures

**Step 4: Commit**

```
git add tests/
git commit -m "test: update integration tests for data_dir + .games refactor"
```
