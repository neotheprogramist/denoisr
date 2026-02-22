"""Binary `.games` format for compact, streaming game storage.

File layout
-----------
[file header: 8 bytes]
    magic:   b"DNSR" (4 bytes)
    version: uint8 (1 byte, currently 1)
    _pad:    3 bytes

[record]*
    [header: 12 bytes]
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
from pathlib import Path
from types import TracebackType
from typing import BinaryIO

from denoisr.types import Action, GameRecord

FILE_MAGIC = b"DNSR"
FORMAT_VERSION = 1

_SENTINEL_ELO = -1
_NO_PROMOTION: int = 0xFF

# Pre-compiled struct formats for performance.
_FILE_HEADER = struct.Struct("<4sB3x")  # magic(4) + version(1) + pad(3) = 8
_RECORD_HEADER = struct.Struct("<fhhBH1x")  # result(4) + w_elo(2) + b_elo(2) + eco_len(1) + num_moves(2) + pad(1) = 12
_MOVE = struct.Struct("<BBB")  # from(1) + to(1) + promo(1) = 3


def write_file_header(fh: BinaryIO) -> None:
    """Write the 8-byte file header to an open binary stream."""
    fh.write(_FILE_HEADER.pack(FILE_MAGIC, FORMAT_VERSION))


def serialize_record(record: GameRecord) -> bytes:
    """Serialize a single GameRecord into a bytes object."""
    eco_bytes = record.eco_code.encode("utf-8") if record.eco_code else b""
    white_elo = record.white_elo if record.white_elo is not None else _SENTINEL_ELO
    black_elo = record.black_elo if record.black_elo is not None else _SENTINEL_ELO

    buf = bytearray()
    buf.extend(
        _RECORD_HEADER.pack(
            record.result,
            white_elo,
            black_elo,
            len(eco_bytes),
            len(record.actions),
        )
    )
    buf.extend(eco_bytes)
    for action in record.actions:
        promo = action.promotion if action.promotion is not None else _NO_PROMOTION
        buf.extend(_MOVE.pack(action.from_square, action.to_square, promo))
    return bytes(buf)


def _read_file_header(fh: BinaryIO) -> None:
    """Read and validate the 8-byte file header."""
    data = fh.read(_FILE_HEADER.size)
    if len(data) < _FILE_HEADER.size:
        raise ValueError("File too short to contain a valid header")
    magic, version = _FILE_HEADER.unpack(data)
    if magic != FILE_MAGIC:
        raise ValueError(
            f"Invalid magic bytes: expected {FILE_MAGIC!r}, got {magic!r}"
        )
    if version != FORMAT_VERSION:
        raise ValueError(
            f"Unsupported format version: expected {FORMAT_VERSION}, got {version}"
        )


def read_game_records(path: Path) -> Iterator[GameRecord]:
    """Yield GameRecord objects from a `.games` file."""
    with open(path, "rb") as fh:
        _read_file_header(fh)
        while True:
            header_data = fh.read(_RECORD_HEADER.size)
            if len(header_data) == 0:
                return
            if len(header_data) < _RECORD_HEADER.size:
                raise ValueError("Truncated record header")
            result, white_elo, black_elo, eco_len, num_moves = (
                _RECORD_HEADER.unpack(header_data)
            )

            eco_code: str | None = None
            if eco_len > 0:
                eco_bytes = fh.read(eco_len)
                if len(eco_bytes) < eco_len:
                    raise ValueError("Truncated ECO code")
                eco_code = eco_bytes.decode("utf-8")

            moves_size = num_moves * _MOVE.size
            moves_data = fh.read(moves_size)
            if len(moves_data) < moves_size:
                raise ValueError("Truncated moves data")

            actions: list[Action] = []
            for i in range(num_moves):
                offset = i * _MOVE.size
                from_sq, to_sq, promo = _MOVE.unpack_from(moves_data, offset)
                promotion: int | None = None if promo == _NO_PROMOTION else promo
                actions.append(Action(from_square=from_sq, to_square=to_sq, promotion=promotion))

            yield GameRecord(
                actions=tuple(actions),
                result=result,
                eco_code=eco_code,
                white_elo=white_elo if white_elo != _SENTINEL_ELO else None,
                black_elo=black_elo if black_elo != _SENTINEL_ELO else None,
            )


def count_positions(path: Path) -> int:
    """Count total moves across all records via header-only scan."""
    total = 0
    with open(path, "rb") as fh:
        _read_file_header(fh)
        while True:
            header_data = fh.read(_RECORD_HEADER.size)
            if len(header_data) == 0:
                return total
            if len(header_data) < _RECORD_HEADER.size:
                raise ValueError("Truncated record header")
            _result, _white_elo, _black_elo, eco_len, num_moves = (
                _RECORD_HEADER.unpack(header_data)
            )
            total += num_moves
            # Skip eco + moves data without reading into Action objects.
            skip_bytes = eco_len + num_moves * _MOVE.size
            fh.seek(skip_bytes, 1)
    return total


class GameBatchWriter:
    """Buffered writer that accumulates records and flushes to disk.

    Flushes when the internal buffer exceeds *max_buffer_bytes*
    (default 16 GiB) or on close.
    """

    def __init__(self, path: Path, max_buffer_bytes: int = 16 * 1024**3) -> None:
        self.path = path
        self.max_buffer_bytes = max_buffer_bytes
        self._buf = bytearray()
        self._count = 0
        self._fh: BinaryIO | None = None

    @property
    def count(self) -> int:
        """Number of records written (buffered + flushed)."""
        return self._count

    def __enter__(self) -> GameBatchWriter:
        self._fh = open(self.path, "wb")  # noqa: SIM115
        write_file_header(self._fh)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def write(self, record: GameRecord) -> None:
        """Buffer a single record. Flushes automatically if buffer is full."""
        self._buf.extend(serialize_record(record))
        self._count += 1
        if len(self._buf) >= self.max_buffer_bytes:
            self.flush()

    def flush(self) -> None:
        """Write the internal buffer to disk and clear it."""
        if self._fh is not None and self._buf:
            self._fh.write(self._buf)
            self._buf.clear()

    def close(self) -> None:
        """Flush remaining data and close the underlying file."""
        self.flush()
        if self._fh is not None:
            self._fh.close()
            self._fh = None
