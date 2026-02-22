import struct

import pytest

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


def _make_record(
    *,
    num_moves: int = 3,
    result: float = 1.0,
    eco_code: str | None = "B90",
    white_elo: int | None = 1500,
    black_elo: int | None = 1600,
    promotion_at: int | None = None,
) -> GameRecord:
    """Helper to build a GameRecord with `num_moves` dummy moves."""
    actions: list[Action] = []
    for i in range(num_moves):
        if promotion_at is not None and i == promotion_at:
            actions.append(Action(from_square=48 + i, to_square=56 + i, promotion=5))
        else:
            actions.append(Action(from_square=i, to_square=i + 8))
    return GameRecord(
        actions=tuple(actions),
        result=result,
        eco_code=eco_code,
        white_elo=white_elo,
        black_elo=black_elo,
    )


class TestSerializeRecord:
    def test_basic_game_round_trip(self) -> None:
        record = _make_record(num_moves=3, white_elo=1500, black_elo=1600, eco_code="B90")
        raw = serialize_record(record)
        # Header (12) + eco "B90" (3) + 3 moves * 3 bytes = 24
        assert len(raw) == 12 + 3 + 9

    def test_no_elo_sentinel(self) -> None:
        record = _make_record(white_elo=None, black_elo=None)
        raw = serialize_record(record)
        # Read white_elo and black_elo from raw bytes at offset 4 and 6
        white_elo_raw = struct.unpack_from("<h", raw, 4)[0]
        black_elo_raw = struct.unpack_from("<h", raw, 6)[0]
        assert white_elo_raw == -1
        assert black_elo_raw == -1

    def test_no_eco_code(self) -> None:
        record = _make_record(eco_code=None)
        raw = serialize_record(record)
        eco_len = struct.unpack_from("<B", raw, 8)[0]
        assert eco_len == 0

    def test_with_promotion(self) -> None:
        record = _make_record(num_moves=3, promotion_at=2)
        raw = serialize_record(record)
        # Promotion move is the 3rd move (index 2)
        # Header is 12 bytes (f=4, h=2, h=2, B=1, H=2, x=1)
        header_size = 12
        eco_len = struct.unpack_from("<B", raw, 8)[0]
        moves_offset = header_size + eco_len
        # 3rd move at moves_offset + 2*3
        promo_offset = moves_offset + 2 * 3
        from_sq, to_sq, promotion = struct.unpack_from("<BBB", raw, promo_offset)
        assert from_sq == 50
        assert to_sq == 58
        assert promotion == 5

    def test_empty_game(self) -> None:
        record = _make_record(num_moves=0, eco_code=None)
        raw = serialize_record(record)
        # 12-byte header, 0 eco, 0 moves
        assert len(raw) == 12


class TestFileIO:
    def test_single_record_round_trip(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "test.games"  # type: ignore[operator]
        record = _make_record()
        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert len(records) == 1
        r = records[0]
        assert r.result == pytest.approx(record.result)
        assert r.white_elo == record.white_elo
        assert r.black_elo == record.black_elo
        assert r.eco_code == record.eco_code
        assert len(r.actions) == len(record.actions)
        for a, b in zip(r.actions, record.actions, strict=True):
            assert a.from_square == b.from_square
            assert a.to_square == b.to_square
            assert a.promotion == b.promotion

    def test_multiple_records(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "multi.games"  # type: ignore[operator]
        records = [_make_record(num_moves=i, result=float(i) / 10) for i in range(10)]
        with open(path, "wb") as fh:
            write_file_header(fh)
            for rec in records:
                fh.write(serialize_record(rec))

        read_back = list(read_game_records(path))
        assert len(read_back) == 10
        for orig, read in zip(records, read_back, strict=True):
            assert len(read.actions) == len(orig.actions)
            assert read.result == pytest.approx(orig.result)

    def test_missing_elo_round_trips_as_none(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "noelo.games"  # type: ignore[operator]
        record = _make_record(white_elo=None, black_elo=None)
        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert records[0].white_elo is None
        assert records[0].black_elo is None

    def test_promotion_round_trips(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "promo.games"  # type: ignore[operator]
        record = _make_record(num_moves=3, promotion_at=1)
        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(record))

        records = list(read_game_records(path))
        assert records[0].actions[1].promotion == 5
        assert records[0].actions[0].promotion is None
        assert records[0].actions[2].promotion is None

    def test_invalid_magic_raises(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "bad.games"  # type: ignore[operator]
        with open(path, "wb") as fh:
            fh.write(b"BAAD\x01\x00\x00\x00")

        with pytest.raises(ValueError, match="magic"):
            list(read_game_records(path))


class TestCountPositions:
    def test_counts_total_moves(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "count.games"  # type: ignore[operator]
        rec3 = _make_record(num_moves=3, eco_code="B90")
        rec1 = _make_record(num_moves=1, eco_code=None)
        with open(path, "wb") as fh:
            write_file_header(fh)
            fh.write(serialize_record(rec3))
            fh.write(serialize_record(rec1))

        assert count_positions(path) == 4


class TestGameBatchWriter:
    def test_flush_on_close(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "batch.games"  # type: ignore[operator]
        with GameBatchWriter(path) as writer:
            for _ in range(5):
                writer.write(_make_record())

        records = list(read_game_records(path))
        assert len(records) == 5

    def test_flush_when_buffer_exceeds_max(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "overflow.games"  # type: ignore[operator]
        with GameBatchWriter(path, max_buffer_bytes=50) as writer:
            for _ in range(20):
                writer.write(_make_record())

        records = list(read_game_records(path))
        assert len(records) == 20

    def test_empty_writer_produces_valid_file(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "empty.games"  # type: ignore[operator]
        with GameBatchWriter(path) as _writer:
            pass

        records = list(read_game_records(path))
        assert len(records) == 0

    def test_count_property(self, tmp_path: pytest.TempPathFactory) -> None:
        path = tmp_path / "counted.games"  # type: ignore[operator]
        with GameBatchWriter(path) as writer:
            assert writer.count == 0
            writer.write(_make_record())
            assert writer.count == 1
            writer.write(_make_record())
            writer.write(_make_record())
            assert writer.count == 3
