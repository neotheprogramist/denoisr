# Binary Game Format + Parallel Sort Design

## Problem

The current `sort_pgn` step is the pipeline's I/O bottleneck:

1. **PGN round-trip waste** — reads GameRecords from `.pgn.zst`, reconstructs full
   `chess.pgn.Game` objects (replaying moves on a Board), serializes to PGN text via
   `str(game)`, zstd-compresses, writes. Later, `generate_data` parses those PGN files
   right back into GameRecords. The text round-trip dominates sort runtime.

2. **Per-game write syscalls** — every single game triggers
   `fh.write(compressor.compress(...))`. Millions of small writes.

3. **Dual data paths** — `generate_data` has two code paths: `_stream_single_pgn()`
   (reads raw PGN with Elo filtering) and `_stream_elo_buckets()` (reads sorted
   `.pgn.zst` files). Redundant logic, extra maintenance surface.

4. **Config path confusion** — three overlapping path concepts: `pgn_path`, `sorted_dir`,
   and `elo_dir` (runtime alias for `sorted_dir`).

## Solution

### 1. Custom binary `.games` format

Replace `.pgn.zst` bucket files with a lightweight binary format that stores exactly
what the pipeline needs — no PGN text reconstruction, no zstd overhead.

**File structure:**

```
[file header: 8 bytes]
  magic:   "DNSR" (4 bytes)
  version: uint8 (1 byte, currently 1)
  _pad:    3 bytes

[record 0]
  [record header: 16 bytes]
    result:    float32 (4 bytes)     — 1.0 / 0.0 / -1.0
    white_elo: int16   (2 bytes)     — -1 = missing
    black_elo: int16   (2 bytes)     — -1 = missing
    eco_len:   uint8   (1 byte)      — 0 = no ECO code
    num_moves: uint16  (2 bytes)
    _pad:      3 bytes
    reserved:  uint16  (2 bytes)

  [eco_code: eco_len bytes, UTF-8]

  [moves: num_moves × 4 bytes]
    from_sq:   uint8 (1 byte, 0-63)
    to_sq:     uint8 (1 byte, 0-63)
    promotion: uint8 (1 byte, 0xFF = none, 2-5 = piece type)
    _pad:      uint8

[record 1]
  ...
```

Typical 40-move game with ECO: 8 (header, first record only) + 16 + 3 + 160 = 179 bytes.
Compare to PGN text (~500-1000 bytes even compressed).

### 2. Unified `data_dir` config

Collapse `pgn_path`, `sorted_dir`, and `elo_dir` into a single `data_dir`:

```toml
[data]
pgn_url = "https://database.lichess.org/..."
data_dir = "data/"
```

Pipeline flow:
- **Fetch** → downloads to `data_dir/raw.pgn.zst`
- **Sort**  → reads `raw.pgn.zst`, writes `data_dir/elo_800.games`, `elo_1200.games`, etc.
- **Generate** → reads `*.games` from `data_dir/`

### 3. Batched writes with 16 GB per-bucket buffers

Each bucket writer accumulates serialized records in an in-memory `bytearray` buffer.
Flush occurs when the buffer exceeds `write_buffer_max_bytes` (default: 16 GB) or on
close. With typical Lichess monthly data (~100M games), most buckets flush once at close.

```python
@dataclass
class GameBatchWriter:
    path: Path
    max_buffer_bytes: int = 16 * 1024**3
    _buffer: bytearray = field(default_factory=bytearray)
    _fh: BinaryIO | None = None
    _count: int = 0

    def write(self, record: GameRecord) -> None:
        self._buffer.extend(_serialize_record(record))
        self._count += 1
        if len(self._buffer) >= self.max_buffer_bytes:
            self.flush()

    def flush(self) -> None:
        if not self._buffer:
            return
        self._fh.write(bytes(self._buffer))
        self._buffer.clear()

    def close(self) -> None:
        self.flush()
        self._fh.close()
```

Context manager `__enter__`/`__exit__` guarantees flush on close.

### 4. Unified position streaming in generate_data

Replace `_stream_single_pgn()` and `_stream_elo_buckets()` with a single
`_stream_game_files()` that reads `.games` binary files:

```python
def _stream_game_files(
    data_dir: Path,
    max_positions: int,
    tactical_fraction: float = 0.25,
    seed: int | None = None,
) -> Iterator[_PositionMeta]:
    game_files = sorted(data_dir.glob("*.games"))
    positions_per_bucket = max_positions // len(game_files)
    for gf in game_files:
        for record in read_game_records(gf):
            # replay moves, yield positions
```

### 5. Fast position counting

Replace `_count_positions()` (which streams and parses full PGN) with a binary
header scanner that reads only record headers from `.games` files to sum
`num_moves` fields. O(N) in records but skips move data entirely.

## Config changes

### DataConfig (pipeline/config.py)

**Remove:** `pgn_path`, `sorted_dir`
**Add:** `data_dir: str = "data/"`, `write_buffer_max_bytes: int = 16 * 1024**3`

### pipeline.toml

```toml
[data]
pgn_url = "https://database.lichess.org/standard/lichess_db_standard_rated_2025-01.pgn.zst"
data_dir = "data/"
stockfish_path = ""
stockfish_depth = 10
examples_per_tier = 2_000_000
tactical_fraction = 0.25
workers = 0
write_buffer_max_bytes = 17_179_869_184  # 16 GB
```

### generate_data.py CLI

**Remove:** `--pgn`, `--elo-dir`
**Add:** `--data-dir` (required, path to directory with `.games` files)

## What gets deleted

- `_stream_single_pgn()` in generate_data.py
- `_stream_elo_buckets()` in generate_data.py (replaced by `_stream_game_files()`)
- `_count_positions()` PGN streaming variant
- PGN reconstruction logic in sort_pgn.py (chess.pgn.Game creation, str(game), zstd)
- `elo_dir` parameter from `generate_to_file()` and `_stream_positions()`
- `pgn_path` from DataConfig
- `sorted_dir` from DataConfig

## New modules

### `src/denoisr/data/game_format.py`

Contains the binary format primitives:
- `MAGIC = b"DNSR"`, `VERSION = 1`
- `write_file_header(fh)` / `read_file_header(fh)`
- `serialize_record(record: GameRecord) -> bytes`
- `read_game_records(path: Path) -> Iterator[GameRecord]`
- `count_positions(path: Path) -> int` (header-only scan)
- `GameBatchWriter` class

### Tests: `tests/test_data/test_game_format.py`

- Round-trip serialization (write + read back)
- Missing Elo handling (-1 sentinel)
- ECO code present/absent
- Promotion encoding (None → 0xFF, 2-5 → piece type)
- Batch writer flush-on-close
- File header magic/version validation
- Fast position counting accuracy

## Affected files

| File | Change |
|------|--------|
| `src/denoisr/data/game_format.py` | **NEW** — binary format + batch writer |
| `src/denoisr/scripts/sort_pgn.py` | Rewrite: binary output instead of PGN |
| `src/denoisr/scripts/generate_data.py` | Replace dual PGN streaming with `.games` reader |
| `src/denoisr/pipeline/config.py` | Remove pgn_path/sorted_dir, add data_dir |
| `src/denoisr/pipeline/steps.py` | Update all steps for data_dir + .games |
| `src/denoisr/scripts/train_phase2.py` | Update PGN streaming to use .games or data_dir |
| `pipeline.toml` | Simplify [data] section |
| `README.md` | Update config examples and CLI docs |
| Tests | Update configs, add game_format tests |
