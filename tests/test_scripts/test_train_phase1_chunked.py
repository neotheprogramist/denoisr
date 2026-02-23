from pathlib import Path

import torch

from denoisr.scripts.train_phase1 import (
    _build_tensor_data_plan,
    _load_examples_from_data,
)


def _make_chunk(path: Path, n: int, start_gid: int = 0) -> None:
    boards = torch.zeros(n, 122, 8, 8, dtype=torch.float32)
    policies = torch.zeros(n, 64, 64, dtype=torch.float32)
    values = torch.zeros(n, 3, dtype=torch.float32)
    values[:, 1] = 1.0  # draw labels
    game_ids = torch.arange(start_gid, start_gid + n, dtype=torch.int64)
    piece_counts = torch.full((n,), 32, dtype=torch.int32)
    torch.save(
        {
            "boards": boards,
            "policies": policies,
            "values": values,
            "game_ids": game_ids,
            "piece_counts": piece_counts,
        },
        path,
    )


def test_load_examples_from_chunked_manifest(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training_data.pt"
    chunk_dir = tmp_path / "training_data_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk0 = chunk_dir / "chunk_000000.pt"
    chunk1 = chunk_dir / "chunk_000001.pt"
    _make_chunk(chunk0, n=2, start_gid=10)
    _make_chunk(chunk1, n=1, start_gid=20)

    torch.save(
        {
            "format": "chunked_v1",
            "total_examples": 3,
            "chunks": [
                {"path": "training_data_chunks/chunk_000000.pt", "count": 2},
                {"path": "training_data_chunks/chunk_000001.pt", "count": 1},
            ],
        },
        manifest_path,
    )

    examples = _load_examples_from_data(manifest_path)
    assert len(examples) == 3
    assert examples[0].game_id == 10
    assert examples[1].game_id == 11
    assert examples[2].game_id == 20


def test_load_examples_from_legacy_single_file(tmp_path: Path) -> None:
    path = tmp_path / "training_data.pt"
    _make_chunk(path, n=2, start_gid=0)
    examples = _load_examples_from_data(path)
    assert len(examples) == 2


def test_build_tensor_data_plan_chunked_counts(tmp_path: Path) -> None:
    manifest_path = tmp_path / "training_data.pt"
    chunk_dir = tmp_path / "training_data_chunks"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    chunk0 = chunk_dir / "chunk_000000.pt"
    chunk1 = chunk_dir / "chunk_000001.pt"
    _make_chunk(chunk0, n=2, start_gid=10)
    _make_chunk(chunk1, n=1, start_gid=20)

    torch.save(
        {
            "format": "chunked_v1",
            "total_examples": 3,
            "chunks": [
                {"path": "training_data_chunks/chunk_000000.pt", "count": 2},
                {"path": "training_data_chunks/chunk_000001.pt", "count": 1},
            ],
        },
        manifest_path,
    )

    plan = _build_tensor_data_plan(manifest_path, holdout_frac=0.5)
    assert plan.total_examples == 3
    assert plan.train_examples == 1
    assert plan.holdout_examples == 2


def test_build_tensor_data_plan_single_file_counts(tmp_path: Path) -> None:
    path = tmp_path / "training_data.pt"
    _make_chunk(path, n=4, start_gid=0)
    plan = _build_tensor_data_plan(path, holdout_frac=0.25)
    assert plan.total_examples == 4
    assert plan.train_examples == 3
    assert plan.holdout_examples == 1


def test_build_tensor_data_plan_grok_tracking_splits(tmp_path: Path) -> None:
    path = tmp_path / "training_data.pt"
    n = 8
    torch.save(
        {
            "boards": torch.zeros(n, 122, 8, 8, dtype=torch.float32),
            "policies": torch.zeros(n, 64, 64, dtype=torch.float32),
            "values": torch.zeros(n, 3, dtype=torch.float32),
            "game_ids": torch.tensor([0, 0, 1, 1, 2, 2, 3, 3], dtype=torch.int64),
            "piece_counts": torch.tensor([5, 32, 32, 32, 32, 32, 32, 32], dtype=torch.int32),
            "eco_codes": ["A00", "A10", "B00", "B10", "C00", "C10", "D00", "D10"],
        },
        path,
    )

    plan = _build_tensor_data_plan(path, holdout_frac=0.25, grok_tracking=True)
    assert plan.total_examples == n

    split_sets = {
        name: set(plan.holdout_split_indices[name][0].tolist())
        for name in ("random", "game_level", "opening_family", "piece_count")
    }
    train_set = set(plan.shards[0].train_indices.tolist())
    holdout_union = set().union(*split_sets.values())

    assert 0 in split_sets["piece_count"]  # Endgame threshold based split.
    assert train_set.isdisjoint(holdout_union)
    assert len(train_set | holdout_union) == n
    assert plan.train_examples == len(train_set)
    for name, idx_set in split_sets.items():
        assert plan.holdout_split_counts[name] == len(idx_set)
