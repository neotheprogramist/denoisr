"""Phase 1: Supervised training from pre-generated training data.

Pipeline:
    training_data.pt -> TrainingExamples -> SupervisedTrainer

Gate to Phase 2: policy top-1 accuracy > 30% on held-out positions.
"""

import math
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.amp import autocast  # type: ignore[attr-defined]
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from denoisr.scripts.config import (
    add_training_args,
    build_backbone,
    build_encoder,
    build_policy_head,
    build_value_head,
    detect_device,
    load_checkpoint,
    maybe_compile,
    resolve_amp_dtype,
    resolve_dataloader_workers,
    save_checkpoint,
    training_config_from_args,
)
from denoisr.scripts.interrupts import graceful_main
from denoisr.scripts.runtime import (
    add_env_argument,
    build_parser,
    configure_logging,
    load_env_file,
)
from denoisr.training.dataset import ChessDataset
from denoisr.training.ema import ModelEMA
from denoisr.training.grok_tracker import GrokTracker
from denoisr.training.grokfast import GrokfastFilter
from denoisr.training.logger import TrainingLogger
from denoisr.training.loss import ChessLossComputer
from denoisr.training.resource_monitor import ResourceMonitor
from denoisr.training.plateau_detector import PlateauDetector
from denoisr.training.prefetch import DevicePrefetcher
from denoisr.training.swa import ModelSWA
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import TrainingExample
from denoisr.types.board import BoardTensor
from denoisr.types.training import PolicyTarget, ValueTarget

log = logging.getLogger(__name__)
_CHUNK_FORMAT = "chunked_v1"
_HOLDOUT_SPLIT_SEED = 42
_GROK_HOLDOUT_SPLITS = ("random", "game_level", "opening_family", "piece_count")
_SWA_START_FRACTION = 0.75
_DATALOADER_PREFETCH_FACTOR = 2
_STABILITY_GUARD_ENABLED = True
_STABILITY_GUARD_MIN_EPOCH = 2
_STABILITY_GUARD_DROP_RATIO = 0.75
_STABILITY_GUARD_MIN_DROP = 0.10
_STABILITY_GUARD_OVERFLOW_FRAC = 5e-4
_STABILITY_GUARD_GRAD_PEAK = 100.0
_STABILITY_GUARD_LR_BACKOFF = 0.5
_STABILITY_GUARD_MAX_BACKOFFS = 4
_STABILITY_GUARD_DISABLE_GROKFAST_AFTER = 1
_STABILITY_GUARD_COOLDOWN_EPOCHS = 1
_OVERFLOW_BACKOFF_MIN_EPOCH = 3
_OVERFLOW_BACKOFF_FRAC = 1e-4
_OVERFLOW_BACKOFF_STREAK = 2
_OVERFLOW_BACKOFF_LR_FACTOR = 0.85
_OVERFLOW_BACKOFF_MAX_BACKOFFS = 4
_OVERFLOW_BACKOFF_COOLDOWN_EPOCHS = 1


@dataclass(frozen=True)
class _TensorDataShard:
    path: Path
    count: int
    train_indices: Tensor
    holdout_indices: Tensor


@dataclass(frozen=True)
class _TensorDataPlan:
    shards: list[_TensorDataShard]
    total_examples: int
    train_examples: int
    holdout_examples: int
    holdout_split_indices: dict[str, list[Tensor]]
    holdout_split_counts: dict[str, int]


@dataclass(frozen=True)
class _StabilityGuardDecision:
    trigger: bool
    reason: str = ""


def _amp_dtype_name(dtype: torch.dtype | None) -> str:
    if dtype is None:
        return "fp32"
    if dtype == torch.bfloat16:
        return "bf16"
    if dtype == torch.float16:
        return "fp16"
    return str(dtype)


class _IndexedTensorDataset(Dataset[tuple[Tensor, Tensor, Tensor]]):
    """Dataset view over selected indices without copying full tensors."""

    def __init__(
        self,
        boards: Tensor,
        policies: Tensor,
        values: Tensor,
        indices: Tensor,
        num_planes: int,
        augment: bool = True,
        value_noise_prob: float = 0.0,
        value_noise_scale: float = 0.02,
        policy_temp_augment_prob: float = 0.0,
    ) -> None:
        self._base = ChessDataset(
            boards=boards,
            policies=policies,
            values=values,
            num_planes=num_planes,
            augment=augment,
            value_noise_prob=value_noise_prob,
            value_noise_scale=value_noise_scale,
            policy_temp_augment_prob=policy_temp_augment_prob,
        )
        self._indices = indices.to(dtype=torch.long, device="cpu")

    def __len__(self) -> int:
        return int(self._indices.numel())

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor, Tensor]:
        base_idx = int(self._indices[idx].item())
        return self._base[base_idx]


def _load_pt_dict(path: Path, *, mmap: bool = False) -> dict[str, Any]:
    load_kwargs: dict[str, Any] = {"weights_only": True}
    if mmap:
        load_kwargs["mmap"] = True
    raw = torch.load(path, **load_kwargs)
    if not isinstance(raw, dict):
        raise ValueError(f"Unexpected training data format at {path}")
    return raw


def _unstack_tensor_dict(data: dict[str, Any]) -> list[TrainingExample]:
    """Convert a stacked tensor dict from .pt file into TrainingExample list."""
    boards = data["boards"]
    policies = data["policies"]
    values = data["values"]
    game_ids = data.get("game_ids")
    eco_codes = data.get("eco_codes")
    piece_counts = data.get("piece_counts")
    n = boards.shape[0]
    return [
        TrainingExample(
            board=BoardTensor(boards[i]),
            policy=PolicyTarget(policies[i]),
            value=ValueTarget(
                win=values[i, 0].item(),
                draw=values[i, 1].item(),
                loss=values[i, 2].item(),
            ),
            game_id=(
                int(game_ids[i].item())
                if game_ids is not None and game_ids[i].item() >= 0
                else None
            ),
            eco_code=(eco_codes[i] if eco_codes is not None else None),
            piece_count=(
                int(piece_counts[i].item())
                if piece_counts is not None and piece_counts[i].item() >= 0
                else None
            ),
        )
        for i in range(n)
    ]


def _load_examples_from_data(path: Path) -> list[TrainingExample]:
    """Load examples from a chunked manifest and its shard files."""
    raw = _load_pt_dict(path, mmap=True)
    if raw.get("format") != _CHUNK_FORMAT:
        raise ValueError(
            f"Unsupported training data format in {path}: expected {_CHUNK_FORMAT!r}"
        )
    chunks = raw.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Chunked manifest has invalid 'chunks' field")
    all_examples: list[TrainingExample] = []
    for idx, chunk_meta in enumerate(chunks):
        if not isinstance(chunk_meta, dict):
            raise ValueError(f"Chunk metadata at index {idx} is invalid")
        rel_path = chunk_meta.get("path")
        if not isinstance(rel_path, str) or rel_path == "":
            raise ValueError(f"Chunk metadata at index {idx} missing path")
        chunk_path = path.parent / rel_path
        chunk_raw = _load_pt_dict(chunk_path, mmap=True)
        all_examples.extend(_unstack_tensor_dict(chunk_raw))
    expected_total = raw.get("total_examples")
    if isinstance(expected_total, int) and expected_total != len(all_examples):
        log.warning(
            "Chunked manifest total_examples=%d but loaded=%d",
            expected_total,
            len(all_examples),
        )
    return all_examples


def _split_indices(
    count: int,
    holdout_frac: float,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    if count <= 0:
        return (
            torch.empty(0, dtype=torch.long),
            torch.empty(0, dtype=torch.long),
        )
    perm = torch.randperm(count, generator=generator)
    holdout_n = max(1, int(count * holdout_frac))
    if count > 1:
        holdout_n = min(holdout_n, count - 1)
    else:
        holdout_n = 1
    holdout_indices = perm[:holdout_n].contiguous()
    train_indices = perm[holdout_n:].contiguous()
    return train_indices, holdout_indices


def _load_shard_payload(shard: _TensorDataShard) -> dict[str, Any]:
    return _load_pt_dict(shard.path, mmap=True)


def _load_shard_tensors(
    shard: _TensorDataShard,
) -> tuple[Tensor, Tensor, Tensor]:
    raw = _load_shard_payload(shard)
    boards = raw.get("boards")
    policies = raw.get("policies")
    values = raw.get("values")
    if not isinstance(boards, torch.Tensor):
        raise ValueError("Tensor data missing 'boards'")
    if not isinstance(policies, torch.Tensor):
        raise ValueError("Tensor data missing 'policies'")
    if not isinstance(values, torch.Tensor):
        raise ValueError("Tensor data missing 'values'")
    return boards, policies, values


def _build_tensor_data_plan(
    path: Path,
    holdout_frac: float,
    *,
    grok_tracking: bool = False,
    endgame_threshold: int = 6,
) -> _TensorDataPlan:
    raw = _load_pt_dict(path, mmap=True)
    if raw.get("format") != _CHUNK_FORMAT:
        raise ValueError(
            f"Unsupported training data format in {path}: expected {_CHUNK_FORMAT!r}"
        )
    g = torch.Generator()
    g.manual_seed(_HOLDOUT_SPLIT_SEED)

    shards: list[_TensorDataShard] = []
    chunks = raw.get("chunks", [])
    if not isinstance(chunks, list):
        raise ValueError("Chunked manifest has invalid 'chunks' field")
    for idx, chunk_meta in enumerate(chunks):
        if not isinstance(chunk_meta, dict):
            raise ValueError(f"Chunk metadata at index {idx} is invalid")
        rel_path = chunk_meta.get("path")
        count = chunk_meta.get("count")
        if not isinstance(rel_path, str) or rel_path == "":
            raise ValueError(f"Chunk metadata at index {idx} missing path")
        if not isinstance(count, int) or count < 0:
            raise ValueError(f"Chunk metadata at index {idx} has invalid count")
        train_idx, holdout_idx = _split_indices(count, holdout_frac, g)
        shards.append(
            _TensorDataShard(
                path=path.parent / rel_path,
                count=count,
                train_indices=train_idx,
                holdout_indices=holdout_idx,
            )
        )

    holdout_split_indices: dict[str, list[Tensor]] = {
        "random": [sh.holdout_indices for sh in shards]
    }
    holdout_split_counts: dict[str, int] = {
        "random": sum(int(sh.holdout_indices.numel()) for sh in shards)
    }

    if grok_tracking:
        total_examples = sum(sh.count for sh in shards)
        game_ids = np.full(total_examples, -1, dtype=np.int64)
        piece_counts = np.full(total_examples, -1, dtype=np.int32)
        eco_families = np.zeros(total_examples, dtype=np.uint8)
        shard_bounds: list[tuple[int, int]] = []

        offset = 0
        for shard in shards:
            start = offset
            end = start + shard.count
            shard_bounds.append((start, end))
            raw_shard = _load_shard_payload(shard)

            game_ids_tensor = raw_shard.get("game_ids")
            if isinstance(game_ids_tensor, torch.Tensor):
                if int(game_ids_tensor.shape[0]) != shard.count:
                    raise ValueError("Tensor data has mismatched 'game_ids' shape")
                game_ids[start:end] = game_ids_tensor.to(
                    dtype=torch.int64, device="cpu"
                ).numpy()

            piece_counts_tensor = raw_shard.get("piece_counts")
            if isinstance(piece_counts_tensor, torch.Tensor):
                if int(piece_counts_tensor.shape[0]) != shard.count:
                    raise ValueError("Tensor data has mismatched 'piece_counts' shape")
                piece_counts[start:end] = piece_counts_tensor.to(
                    dtype=torch.int32, device="cpu"
                ).numpy()

            eco_codes = raw_shard.get("eco_codes")
            if isinstance(eco_codes, list):
                if len(eco_codes) != shard.count:
                    raise ValueError("Tensor data has mismatched 'eco_codes' size")
                for local_idx, eco in enumerate(eco_codes):
                    if isinstance(eco, str) and eco:
                        eco_families[start + local_idx] = ord(eco[0].upper())
            offset = end

        rng = random.Random(_HOLDOUT_SPLIT_SEED)
        excluded = np.zeros(total_examples, dtype=bool)
        split_masks: dict[str, np.ndarray] = {}

        piece_mask = (piece_counts >= 0) & (piece_counts <= endgame_threshold)
        split_masks["piece_count"] = piece_mask.copy()
        excluded |= piece_mask

        unique_game_ids = [int(gid) for gid in np.unique(game_ids) if gid >= 0]
        game_level_mask = np.zeros(total_examples, dtype=bool)
        if len(unique_game_ids) >= 2:
            holdout_game_count = max(1, int(len(unique_game_ids) * holdout_frac))
            holdout_game_count = min(holdout_game_count, len(unique_game_ids) - 1)
            holdout_game_ids = set(
                rng.sample(sorted(unique_game_ids), holdout_game_count)
            )
            game_level_mask = np.isin(
                game_ids, np.array(list(holdout_game_ids), dtype=np.int64)
            )
            game_level_mask &= ~excluded
        split_masks["game_level"] = game_level_mask
        excluded |= game_level_mask

        unique_families = [int(code) for code in np.unique(eco_families) if code > 0]
        opening_family_mask = np.zeros(total_examples, dtype=bool)
        if len(unique_families) >= 2:
            holdout_family_count = max(1, int(len(unique_families) * holdout_frac))
            holdout_family_count = min(holdout_family_count, len(unique_families) - 1)
            holdout_families = set(
                rng.sample(sorted(unique_families), holdout_family_count)
            )
            opening_family_mask = np.isin(
                eco_families,
                np.array(list(holdout_families), dtype=np.uint8),
            )
            opening_family_mask &= ~excluded
        split_masks["opening_family"] = opening_family_mask
        excluded |= opening_family_mask

        holdout_n = max(1, int(total_examples * holdout_frac))
        remaining_indices = np.flatnonzero(~excluded)
        random_holdout_n = min(holdout_n, int(remaining_indices.size))
        random_mask = np.zeros(total_examples, dtype=bool)
        if random_holdout_n > 0:
            random_holdout_indices = rng.sample(
                remaining_indices.tolist(),
                random_holdout_n,
            )
            random_mask[np.asarray(random_holdout_indices, dtype=np.int64)] = True
        split_masks["random"] = random_mask
        excluded |= random_mask

        train_mask = ~excluded

        def _mask_to_shard_indices(mask: np.ndarray) -> list[Tensor]:
            result: list[Tensor] = []
            for start, end in shard_bounds:
                local_indices = np.flatnonzero(mask[start:end]).astype(np.int64)
                if local_indices.size == 0:
                    result.append(torch.empty(0, dtype=torch.long))
                else:
                    result.append(torch.from_numpy(local_indices.copy()))
            return result

        split_indices = {
            name: _mask_to_shard_indices(mask) for name, mask in split_masks.items()
        }
        train_indices = _mask_to_shard_indices(train_mask)
        holdout_split_indices = {
            name: split_indices[name] for name in _GROK_HOLDOUT_SPLITS
        }
        holdout_split_counts = {
            name: int(split_masks[name].sum()) for name in _GROK_HOLDOUT_SPLITS
        }
        shards = [
            _TensorDataShard(
                path=sh.path,
                count=sh.count,
                train_indices=train_indices[i],
                holdout_indices=split_indices["random"][i],
            )
            for i, sh in enumerate(shards)
        ]

    total_examples = sum(sh.count for sh in shards)
    train_examples = sum(int(sh.train_indices.numel()) for sh in shards)
    holdout_examples = holdout_split_counts.get("random", 0)
    return _TensorDataPlan(
        shards=shards,
        total_examples=total_examples,
        train_examples=train_examples,
        holdout_examples=holdout_examples,
        holdout_split_indices=holdout_split_indices,
        holdout_split_counts=holdout_split_counts,
    )


def _measure_accuracy_from_indices(
    trainer: SupervisedTrainer,
    data_plan: _TensorDataPlan,
    indices_by_shard: list[Tensor],
    device: torch.device,
    batch_size: int = 256,
    use_tqdm: bool = False,
    progress_desc: str | None = None,
) -> tuple[float, float]:
    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    amp_dtype: torch.dtype | None = getattr(trainer, "_amp_dtype", None)
    autocast_enabled = amp_dtype is not None and device.type == "cuda"

    correct_1 = 0
    correct_5 = 0
    total = 0
    total_eval = sum(int(idx.numel()) for idx in indices_by_shard)
    eval_bar = tqdm(
        total=total_eval,
        desc=progress_desc or "Evaluating holdout",
        unit="pos",
        leave=False,
        disable=(not use_tqdm) or (total_eval <= 0),
        smoothing=0.3,
    )

    with torch.no_grad(), autocast(autocast_device, enabled=autocast_enabled, dtype=amp_dtype):
        for shard_idx, shard in enumerate(data_plan.shards):
            split_idx = indices_by_shard[shard_idx]
            if split_idx.numel() == 0:
                continue
            boards, policies, _values = _load_shard_tensors(shard)
            n = int(split_idx.numel())
            total += n
            for i in range(0, n, batch_size):
                batch_idx = split_idx[i : i + batch_size]
                boards_batch = boards[batch_idx].to(device, non_blocking=True)
                target_batch = policies[batch_idx].to(device, non_blocking=True)

                latent = trainer.encoder(boards_batch)
                features = trainer.backbone(latent)
                logits = trainer.policy_head(features)

                pred_flat = logits.reshape(len(batch_idx), -1)
                target_flat = target_batch.reshape(len(batch_idx), -1)
                legal_mask = target_flat > 0
                masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
                target_idx = target_flat.argmax(dim=-1)

                top5 = masked_logits.topk(5, dim=-1).indices
                correct_1 += (top5[:, 0] == target_idx).sum().item()
                correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()
                eval_bar.update(int(batch_idx.numel()))
    eval_bar.close()

    return correct_1 / max(total, 1), correct_5 / max(total, 1)


def _measure_accuracy_from_plan(
    trainer: SupervisedTrainer,
    data_plan: _TensorDataPlan,
    device: torch.device,
    batch_size: int = 256,
    use_tqdm: bool = False,
    progress_desc: str | None = None,
) -> tuple[float, float]:
    return _measure_accuracy_from_indices(
        trainer=trainer,
        data_plan=data_plan,
        indices_by_shard=[sh.holdout_indices for sh in data_plan.shards],
        device=device,
        batch_size=batch_size,
        use_tqdm=use_tqdm,
        progress_desc=progress_desc,
    )


def measure_accuracy(
    trainer: SupervisedTrainer,
    examples: list[TrainingExample],
    device: torch.device,
    batch_size: int = 256,
) -> tuple[float, float]:
    trainer.encoder.eval()
    trainer.backbone.eval()
    trainer.policy_head.eval()

    autocast_device = device.type if device.type in ("cuda", "cpu") else "cpu"
    amp_dtype_val: torch.dtype | None = getattr(trainer, "_amp_dtype", None)
    autocast_enabled = amp_dtype_val is not None and device.type == "cuda"

    correct_1 = 0
    correct_5 = 0
    total = len(examples)

    with torch.no_grad(), autocast(autocast_device, enabled=autocast_enabled, dtype=amp_dtype_val):
        for i in range(0, total, batch_size):
            batch = examples[i : i + batch_size]
            boards = torch.stack([ex.board.data for ex in batch]).to(device)
            targets = torch.stack([ex.policy.data for ex in batch]).to(device)

            latent = trainer.encoder(boards)
            features = trainer.backbone(latent)
            logits = trainer.policy_head(features)

            pred_flat = logits.reshape(len(batch), -1)
            target_flat = targets.reshape(len(batch), -1)
            legal_mask = target_flat > 0
            masked_logits = pred_flat.masked_fill(~legal_mask, float("-inf"))
            target_idx = target_flat.argmax(dim=-1)  # (B,)

            top5 = masked_logits.topk(5, dim=-1).indices  # (B, 5)
            correct_1 += (top5[:, 0] == target_idx).sum().item()
            correct_5 += (top5 == target_idx.unsqueeze(1)).any(dim=1).sum().item()

    return correct_1 / max(total, 1), correct_5 / max(total, 1)


def _is_dataloader_worker_failure(exc: RuntimeError) -> bool:
    msg = str(exc)
    return "DataLoader worker" in msg and "exited unexpectedly" in msg


def _is_loss_stall_warning(msg: str) -> bool:
    return "loss stalled" in msg


def _build_phase1_train_loader(
    train_dataset: Dataset[tuple[Tensor, Tensor, Tensor]],
    *,
    batch_size: int,
    worker_count: int,
    pin_memory: bool,
) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
    loader_kwargs: dict[str, Any] = {
        "batch_size": batch_size,
        "shuffle": True,
        "num_workers": worker_count,
        "pin_memory": pin_memory,
    }
    if worker_count > 0:
        loader_kwargs["prefetch_factor"] = _DATALOADER_PREFETCH_FACTOR
    return DataLoader(train_dataset, **loader_kwargs)


def _evaluate_stability_guard(
    *,
    enabled: bool,
    epoch_num: int,
    best_top1: float,
    current_top1: float,
    overflow_frac: float,
    grad_peak: float,
    backoff_count: int,
    cooldown_until_epoch: int,
    min_epoch: int,
    drop_ratio: float,
    min_drop: float,
    overflow_frac_threshold: float,
    grad_peak_threshold: float,
    max_backoffs: int,
) -> _StabilityGuardDecision:
    if not enabled:
        return _StabilityGuardDecision(trigger=False)
    if best_top1 <= 0.0:
        return _StabilityGuardDecision(trigger=False)
    if epoch_num < min_epoch:
        return _StabilityGuardDecision(trigger=False)
    if backoff_count >= max_backoffs:
        return _StabilityGuardDecision(trigger=False)
    if epoch_num <= cooldown_until_epoch:
        return _StabilityGuardDecision(trigger=False)

    drop = best_top1 - current_top1
    if drop < min_drop:
        return _StabilityGuardDecision(trigger=False)

    ratio = current_top1 / max(best_top1, 1e-12)
    if ratio > drop_ratio:
        return _StabilityGuardDecision(trigger=False)

    overflow_signal = overflow_frac >= overflow_frac_threshold
    grad_signal = math.isfinite(grad_peak) and grad_peak >= grad_peak_threshold
    if not (overflow_signal or grad_signal):
        return _StabilityGuardDecision(trigger=False)

    reasons: list[str] = [
        (
            f"top1 dropped {drop * 100:.2f}pp "
            f"(best={best_top1 * 100:.2f}%, now={current_top1 * 100:.2f}%)"
        )
    ]
    if overflow_signal:
        reasons.append(f"overflow_frac={overflow_frac * 100:.3f}%")
    if grad_signal:
        reasons.append(f"grad_peak={grad_peak:.1f}")
    return _StabilityGuardDecision(trigger=True, reason="; ".join(reasons))


def _restore_phase1_from_checkpoint(
    *,
    checkpoint_path: Path,
    device: torch.device,
    trainer: SupervisedTrainer,
    model_ema: ModelEMA | None,
) -> bool:
    if not checkpoint_path.exists():
        return False
    _ckpt_cfg, restored_state = load_checkpoint(checkpoint_path, device)
    trainer.encoder.load_state_dict(restored_state["encoder"])
    trainer.backbone.load_state_dict(restored_state["backbone"])
    trainer.policy_head.load_state_dict(restored_state["policy_head"])
    trainer.value_head.load_state_dict(restored_state["value_head"])

    optimizer_state = restored_state.get("optimizer")
    if isinstance(optimizer_state, dict):
        trainer.optimizer.load_state_dict(optimizer_state)

    if model_ema is not None:
        ema_state: dict[str, dict[str, Tensor]] = {}
        for name in ("encoder", "backbone", "policy_head", "value_head"):
            key = f"ema_{name}"
            state_dict = restored_state.get(key)
            if isinstance(state_dict, dict):
                ema_state[name] = state_dict
        if len(ema_state) == 4:
            model_ema.load_state_dicts(ema_state)
    return True


@graceful_main("denoisr-train-phase1", logger=log)
def main() -> None:
    load_env_file()
    parser = build_parser("Phase 1: Supervised training")
    add_env_argument(
        parser,
        "--checkpoint",
        env_var="DENOISR_PHASE1_CHECKPOINT",
        help="Checkpoint to load (create with denoisr-init)",
    )
    add_env_argument(
        parser,
        "--data",
        env_var="DENOISR_PHASE1_DATA",
        help="Path to training data .pt file (create with denoisr-generate-data)",
    )
    add_env_argument(
        parser,
        "--holdout-frac",
        env_var="DENOISR_PHASE1_HOLDOUT_FRAC",
        type=float,
    )
    add_env_argument(
        parser,
        "--batch-size",
        env_var="DENOISR_PHASE1_BATCH_SIZE",
        type=int,
    )
    add_env_argument(
        parser,
        "--epochs",
        env_var="DENOISR_PHASE1_EPOCHS",
        type=int,
    )
    add_env_argument(
        parser,
        "--lr",
        env_var="DENOISR_PHASE1_LR",
        type=float,
    )
    add_env_argument(
        parser,
        "--output",
        env_var="DENOISR_PHASE1_OUTPUT",
        type=str,
    )
    add_env_argument(
        parser,
        "--run-name",
        env_var="DENOISR_RUN_NAME",
        type=str,
        default=None,
        required=False,
        help="log run label attached to training log lines (default: timestamp)",
    )
    add_training_args(parser)
    args = parser.parse_args()

    log_path = configure_logging()
    log.info("logging to %s", log_path)

    device = detect_device()
    tcfg = training_config_from_args(args)
    if tcfg.grokfast_start_epoch < 1:
        raise ValueError("--grokfast-start-epoch must be >= 1")
    if tcfg.grokfast_plateau_epochs < 0:
        raise ValueError("--grokfast-plateau-epochs must be >= 0")
    if tcfg.phase1_ema_eval_every < 1:
        raise ValueError("--phase1-ema-eval-every must be >= 1")
    if tcfg.phase1_swa_eval_every < 1:
        raise ValueError("--phase1-swa-eval-every must be >= 1")
    if tcfg.phase1_grok_eval_every < 1:
        raise ValueError("--phase1-grok-eval-every must be >= 1")
    use_tqdm = args.tqdm
    log.info("device=%s", device)

    # --- Load checkpoint ---
    cfg, state = load_checkpoint(Path(args.checkpoint), device)
    log.info(
        "checkpoint loaded  d_s=%d  heads=%d  layers=%d",
        cfg.d_s,
        cfg.num_heads,
        cfg.num_layers,
    )

    encoder = build_encoder(cfg).to(device)
    backbone = build_backbone(cfg).to(device)
    policy_head = build_policy_head(cfg).to(device)
    value_head = build_value_head(cfg).to(device)

    encoder.load_state_dict(state["encoder"])
    backbone.load_state_dict(state["backbone"])
    policy_head.load_state_dict(state["policy_head"])
    value_head.load_state_dict(state["value_head"])

    encoder = maybe_compile(encoder, device)
    backbone = maybe_compile(backbone, device)
    policy_head = maybe_compile(policy_head, device)
    value_head = maybe_compile(value_head, device)

    # --- Load pre-generated data (mmap + shard-index split to bound RAM) ---
    data_plan = _build_tensor_data_plan(
        Path(args.data),
        args.holdout_frac,
        grok_tracking=tcfg.grok_tracking,
        endgame_threshold=6,
    )
    if data_plan.total_examples <= 0:
        raise ValueError(f"No examples available in training data: {args.data}")
    if data_plan.train_examples <= 0:
        raise ValueError(
            "No train examples available after holdout split; "
            "reduce --holdout-frac or regenerate data with more examples."
        )
    log.info("examples=%d  source=%s", data_plan.total_examples, args.data)
    split_counts_msg = ", ".join(
        f"{name}={data_plan.holdout_split_counts.get(name, 0)}"
        for name in _GROK_HOLDOUT_SPLITS
        if name in data_plan.holdout_split_counts
    )
    log.info("train=%d  holdout splits: %s", data_plan.train_examples, split_counts_msg)

    loss_fn = ChessLossComputer(
        policy_weight=tcfg.policy_weight,
        value_weight=tcfg.value_weight,
        use_harmony_dream=tcfg.use_harmony_dream,
        harmony_ema_decay=tcfg.harmony_ema_decay,
        illegal_penalty_weight=tcfg.illegal_penalty_weight,
        label_smoothing=tcfg.label_smoothing,
    )

    # --- Grokfast filter (opt-in) ---
    grokfast_filter: GrokfastFilter | None = None
    grokfast_enabled = False
    if tcfg.grokfast:
        immediate_grokfast = (
            tcfg.grokfast_start_epoch <= 1 and tcfg.grokfast_plateau_epochs == 0
        )
        if immediate_grokfast:
            grokfast_filter = GrokfastFilter(
                alpha=tcfg.grokfast_alpha,
                lamb=tcfg.grokfast_lamb,
            )
            grokfast_enabled = True
            log.info(
                "grokfast enabled  alpha=%.3f  lamb=%.1f",
                tcfg.grokfast_alpha,
                tcfg.grokfast_lamb,
            )
        else:
            log.info(
                (
                    "grokfast delayed  alpha=%.3f  lamb=%.1f  "
                    "start_epoch=%d  plateau_epochs=%d"
                ),
                tcfg.grokfast_alpha,
                tcfg.grokfast_lamb,
                tcfg.grokfast_start_epoch,
                tcfg.grokfast_plateau_epochs,
            )

    # --- EMA shadow model (opt-in) ---
    model_ema: ModelEMA | None = None
    if tcfg.ema_decay > 0:
        model_ema = ModelEMA(
            {
                "encoder": encoder,
                "backbone": backbone,
                "policy_head": policy_head,
                "value_head": value_head,
            },
            decay=tcfg.ema_decay,
        )
        log.info("EMA enabled  decay=%.4f", tcfg.ema_decay)

    # Compute actual micro-batches: each shard's DataLoader rounds up
    # independently (drop_last=False), so use ceiling division per shard.
    steps_per_epoch = sum(
        -(-int(shard.train_indices.numel()) // args.batch_size)
        for shard in data_plan.shards
        if shard.train_indices.numel() > 0
    )
    requested_amp_dtype = resolve_amp_dtype(tcfg)
    active_amp_dtype = requested_amp_dtype
    if device.type == "cuda" and active_amp_dtype != torch.bfloat16:
        if torch.cuda.is_bf16_supported():
            log.warning(
                "Overriding amp_dtype=%s to bf16 on CUDA to minimize overflow risk.",
                _amp_dtype_name(active_amp_dtype),
            )
            active_amp_dtype = torch.bfloat16
        else:
            log.warning(
                (
                    "CUDA device reports bf16 unsupported; keeping amp_dtype=%s. "
                    "Overflow risk may be higher."
                ),
                _amp_dtype_name(active_amp_dtype),
            )
    trainer = SupervisedTrainer(
        encoder=encoder,
        backbone=backbone,
        policy_head=policy_head,
        value_head=value_head,
        loss_fn=loss_fn,
        lr=args.lr,
        device=device,
        total_epochs=args.epochs,
        warmup_epochs=tcfg.warmup_epochs,
        max_grad_norm=tcfg.max_grad_norm,
        weight_decay=tcfg.weight_decay,
        encoder_lr_multiplier=tcfg.encoder_lr_multiplier,
        min_lr=tcfg.min_lr,
        grokfast_filter=grokfast_filter,
        use_warm_restarts=tcfg.use_warm_restarts,
        use_onecycle=tcfg.use_onecycle,
        onecycle_pct_start=tcfg.onecycle_pct_start,
        steps_per_epoch=steps_per_epoch,
        accumulation_steps=tcfg.gradient_accumulation_steps,
        amp_dtype=active_amp_dtype,
    )
    log.info(
        "AMP config: requested=%s active=%s autocast=%s grad_scaler=%s",
        _amp_dtype_name(requested_amp_dtype),
        _amp_dtype_name(trainer.amp_dtype),
        trainer.amp_autocast_enabled,
        trainer.amp_scaler_enabled,
    )
    swa_start_epoch = max(1, math.ceil(args.epochs * _SWA_START_FRACTION))
    swa_model = ModelSWA(
        {
            "encoder": encoder,
            "backbone": backbone,
            "policy_head": policy_head,
            "value_head": value_head,
        }
    )
    log.info(
        "SWA enabled  start_epoch=%d/%d",
        swa_start_epoch,
        args.epochs,
    )
    if not swa_model.has_batch_norm():
        log.info("SWA batch-norm update skipped (no BatchNorm layers detected)")

    # --- Grok tracker (opt-in) ---
    grok_tracker: GrokTracker | None = None

    monitor = ResourceMonitor()
    plateau_detector = PlateauDetector(warmup_epochs=tcfg.warmup_epochs)

    with TrainingLogger(Path("logs"), run_name=args.run_name) as logger:
        if tcfg.grok_tracking:
            grok_tracker = GrokTracker(
                encoder=encoder,
                backbone=backbone,
                policy_head=policy_head,
                value_head=value_head,
                erank_freq=tcfg.grok_erank_freq,
                spectral_freq=tcfg.grok_spectral_freq,
                onset_threshold=tcfg.grok_onset_threshold,
                on_state_transition=logger.log_grok_state_transition,
            )
            log.info(
                "grok tracking enabled  erank_freq=%d  spectral_freq=%d",
                tcfg.grok_erank_freq,
                tcfg.grok_spectral_freq,
            )
        # --- Build shard stream config ---
        bs = args.batch_size
        worker_count = resolve_dataloader_workers(tcfg.workers)
        active_worker_count = worker_count
        pin_memory = device.type == "cuda"
        log.info(
            "phase1 dataloader config: workers=%d  batch_size=%d  shards=%d",
            active_worker_count,
            bs,
            len(data_plan.shards),
        )
        log.info(
            (
                "phase1 eval cadence: ema_every=%d  swa_every=%d  "
                "grok_nonrandom_every=%d"
            ),
            tcfg.phase1_ema_eval_every,
            tcfg.phase1_swa_eval_every,
            tcfg.phase1_grok_eval_every,
        )

        # --- Train ---
        best_acc = 0.0
        best_source = "base"

        logger.log_hparams(
            {
                "lr": args.lr,
                "batch_size": bs,
                "epochs": args.epochs,
                "d_s": cfg.d_s,
                "num_heads": cfg.num_heads,
                "num_layers": cfg.num_layers,
                "ffn_dim": cfg.ffn_dim,
                "num_planes": cfg.num_planes,
                "gradient_checkpointing": cfg.gradient_checkpointing,
                "max_grad_norm": tcfg.max_grad_norm,
                "weight_decay": tcfg.weight_decay,
                "encoder_lr_multiplier": tcfg.encoder_lr_multiplier,
                "min_lr": tcfg.min_lr,
                "warmup_epochs": tcfg.warmup_epochs,
                "amp_dtype": _amp_dtype_name(trainer.amp_dtype),
                "policy_weight": tcfg.policy_weight,
                "value_weight": tcfg.value_weight,
                "use_harmony_dream": tcfg.use_harmony_dream,
                "workers": worker_count,
                "grokfast_lamb": tcfg.grokfast_lamb,
                "grokfast_start_epoch": tcfg.grokfast_start_epoch,
                "grokfast_plateau_epochs": tcfg.grokfast_plateau_epochs,
                "phase1_ema_eval_every": tcfg.phase1_ema_eval_every,
                "phase1_swa_eval_every": tcfg.phase1_swa_eval_every,
                "phase1_grok_eval_every": tcfg.phase1_grok_eval_every,
            },
            {"best_top1": 0.0},
        )

        global_step = 0
        consecutive_overflow_epochs = 0
        overflow_backoff_streak = 0
        overflow_backoff_count = 0
        overflow_backoff_cooldown_until_epoch = 0
        stability_backoff_count = 0
        stability_guard_cooldown_until_epoch = 0
        grokfast_disabled_by_guard = False
        grokfast_plateau_streak = 0

        def _save_current_checkpoint() -> None:
            ema_kwargs: dict[str, object] = {}
            if model_ema is not None:
                for name, sd in model_ema.state_dicts().items():
                    ema_kwargs[f"ema_{name}"] = sd
            save_checkpoint(
                Path(args.output),
                cfg,
                encoder=encoder.state_dict(),
                backbone=backbone.state_dict(),
                policy_head=policy_head.state_dict(),
                value_head=value_head.state_dict(),
                optimizer=trainer.optimizer.state_dict(),
                **ema_kwargs,
            )

        for epoch in range(args.epochs):
            epoch_num = epoch + 1
            if (
                tcfg.grokfast
                and not grokfast_enabled
                and not grokfast_disabled_by_guard
                and epoch_num >= tcfg.grokfast_start_epoch
                and grokfast_plateau_streak >= tcfg.grokfast_plateau_epochs
            ):
                trainer.set_grokfast_filter(
                    GrokfastFilter(
                        alpha=tcfg.grokfast_alpha,
                        lamb=tcfg.grokfast_lamb,
                    )
                )
                grokfast_enabled = True
                log.info(
                    (
                        "grokfast enabled at epoch %d  alpha=%.3f  lamb=%.1f  "
                        "plateau_streak=%d/%d"
                    ),
                    epoch_num,
                    tcfg.grokfast_alpha,
                    tcfg.grokfast_lamb,
                    grokfast_plateau_streak,
                    tcfg.grokfast_plateau_epochs,
                )
            log.info("Epoch %d/%d started", epoch_num, args.epochs)
            epoch_loss = 0.0
            num_batches = 0
            num_loss_batches = 0
            epoch_start = time.monotonic()
            monitor.reset()
            step_grad_norms: list[float] = []
            policy_loss_sum = 0.0
            value_loss_sum = 0.0
            policy_loss_count = 0
            value_loss_count = 0
            overflow_count = 0
            data_time = 0.0

            shard_order = torch.randperm(len(data_plan.shards)).tolist()
            for shard_pos, shard_idx in enumerate(shard_order, start=1):
                shard = data_plan.shards[shard_idx]
                if shard.train_indices.numel() == 0:
                    continue
                boards, policies, values = _load_shard_tensors(shard)
                train_dataset = _IndexedTensorDataset(
                    boards=boards,
                    policies=policies,
                    values=values,
                    indices=shard.train_indices,
                    num_planes=cfg.num_planes,
                    augment=True,
                    value_noise_prob=tcfg.value_noise_prob,
                    value_noise_scale=tcfg.value_noise_scale,
                    policy_temp_augment_prob=tcfg.policy_temp_augment_prob,
                )
                train_loader = _build_phase1_train_loader(
                    train_dataset,
                    batch_size=bs,
                    worker_count=active_worker_count,
                    pin_memory=pin_memory,
                )
                batch_iter: (
                    DevicePrefetcher[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                    | DataLoader[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
                )
                if device.type == "cuda":
                    try:
                        batch_iter = DevicePrefetcher(train_loader, device)
                    except RuntimeError as exc:
                        if (
                            active_worker_count > 0
                            and _is_dataloader_worker_failure(exc)
                        ):
                            log.warning(
                                "DataLoader worker startup failed on shard %d/%d; "
                                "disabling workers (num_workers=0) for the "
                                "remainder of phase1 "
                                "(%s)",
                                shard_pos,
                                len(shard_order),
                                exc,
                            )
                            active_worker_count = 0
                            del train_loader
                            train_loader = _build_phase1_train_loader(
                                train_dataset,
                                batch_size=bs,
                                worker_count=active_worker_count,
                                pin_memory=pin_memory,
                            )
                            batch_iter = DevicePrefetcher(train_loader, device)
                        else:
                            raise
                else:
                    batch_iter = train_loader
                pbar = tqdm(
                    batch_iter,
                    desc=(
                        f"Epoch {epoch_num}/{args.epochs} "
                        f"[shard {shard_pos}/{len(shard_order)}]"
                    ),
                    total=len(train_loader),
                    leave=False,
                    smoothing=0.3,
                    disable=not use_tqdm,
                )
                data_start = time.monotonic()
                for boards_batch, policies_batch, values_batch in pbar:
                    data_time += time.monotonic() - data_start

                    loss, breakdown = trainer.train_step_tensors(
                        boards_batch, policies_batch, values_batch
                    )
                    if model_ema is not None:
                        model_ema.update()

                    if grok_tracker is not None:
                        _ = grok_tracker.step(
                            global_step, breakdown, breakdown.get("grad_norm", 0.0)
                        )
                    if math.isfinite(loss):
                        epoch_loss += loss
                        num_loss_batches += 1
                    policy_loss = float(breakdown.get("policy", 0.0))
                    if math.isfinite(policy_loss):
                        policy_loss_sum += policy_loss
                        policy_loss_count += 1
                    value_loss = float(breakdown.get("value", 0.0))
                    if math.isfinite(value_loss):
                        value_loss_sum += value_loss
                        value_loss_count += 1
                    if breakdown.get("overflow", False):
                        overflow_count += 1
                    else:
                        step_grad_norms.append(breakdown.get("grad_norm", 0.0))
                    if global_step % 100 == 0:
                        monitor.sample()
                    global_step += 1
                    num_batches += 1
                    if use_tqdm:
                        pbar.set_postfix(
                            loss=f"{loss:.4f}",
                            policy=f"{breakdown['policy']:.4f}",
                            value=f"{breakdown['value']:.4f}",
                        )
                    data_start = time.monotonic()
                pbar.close()
                del train_loader, train_dataset, boards, policies, values
            trainer.scheduler_step()
            if epoch_num >= swa_start_epoch:
                swa_model.update()

            epoch_duration = time.monotonic() - epoch_start
            samples_per_sec = data_plan.train_examples / max(epoch_duration, 1e-9)
            avg_loss = epoch_loss / max(num_loss_batches, 1)
            top1, top5 = _measure_accuracy_from_plan(
                trainer,
                data_plan,
                device,
                use_tqdm=use_tqdm,
                progress_desc=f"Eval random E{epoch_num}",
            )
            boundary_epoch = epoch_num in {1, args.epochs}
            swa_top1: float | None = None
            swa_top5: float | None = None
            eval_swa = (
                swa_model.num_updates > 0
                and (boundary_epoch or (epoch_num % tcfg.phase1_swa_eval_every == 0))
            )
            if eval_swa:
                with swa_model.apply():
                    swa_top1, swa_top5 = _measure_accuracy_from_plan(
                        trainer,
                        data_plan,
                        device,
                        use_tqdm=use_tqdm,
                        progress_desc=f"Eval random SWA E{epoch_num}",
                    )
                log.info(
                    "SWA top1=%.2f%% top5=%.2f%% (regular=%.2f%%)",
                    swa_top1 * 100,
                    (swa_top5 or 0.0) * 100,
                    top1 * 100,
                )
            ema_top1: float | None = None
            eval_ema = model_ema is not None and (
                boundary_epoch or (epoch_num % tcfg.phase1_ema_eval_every == 0)
            )
            if eval_ema and model_ema is not None:
                with model_ema.apply():
                    ema_top1, _ = _measure_accuracy_from_plan(
                        trainer,
                        data_plan,
                        device,
                        use_tqdm=use_tqdm,
                        progress_desc=f"Eval random EMA E{epoch_num}",
                    )
                log.info(
                    "EMA top1=%.2f%% (regular=%.2f%%)",
                    ema_top1 * 100,
                    top1 * 100,
                )
            selected_top1 = top1
            selected_source = "base"
            if ema_top1 is not None and ema_top1 >= selected_top1:
                selected_top1 = ema_top1
                selected_source = "ema"
            if swa_top1 is not None and swa_top1 >= selected_top1:
                selected_top1 = swa_top1
                selected_source = "swa"
            current_lr = trainer.optimizer.param_groups[0]["lr"]
            avg_policy_loss = policy_loss_sum / max(policy_loss_count, 1)
            avg_value_loss = value_loss_sum / max(value_loss_count, 1)
            overflow_frac = overflow_count / max(num_batches, 1)

            skipped_loss_batches = num_batches - num_loss_batches
            if skipped_loss_batches > 0:
                log.warning(
                    "Epoch %d skipped %d non-finite batches in loss aggregates.",
                    epoch + 1,
                    skipped_loss_batches,
                )

            if overflow_frac > 0.25:
                log.warning(
                    (
                        "High overflow rate in epoch %d: %.1f%% of batches "
                        "(ovf=%d/%d). Training may stall; consider lowering "
                        "lr or disabling/tuning Grokfast."
                    ),
                    epoch + 1,
                    overflow_frac * 100.0,
                    overflow_count,
                    num_batches,
                )
            if overflow_count > 0 and overflow_frac >= _OVERFLOW_BACKOFF_FRAC:
                overflow_backoff_streak += 1
            else:
                overflow_backoff_streak = 0
            if overflow_count >= num_batches and num_batches > 0:
                consecutive_overflow_epochs += 1
            else:
                consecutive_overflow_epochs = 0
            stop_for_overflow = False
            if consecutive_overflow_epochs >= 2:
                log.error(
                    (
                        "All batches overflowed for %d consecutive epochs. "
                        "Stopping early to avoid wasted compute."
                    ),
                    consecutive_overflow_epochs,
                )
                stop_for_overflow = True

            resource_metrics = monitor.summarize()
            resources: dict[str, str] | None = None
            if resource_metrics:
                resources = {
                    "cpu_pct": f"{resource_metrics['cpu_percent_avg']:.0f}%",
                    "cpu_max": f"{resource_metrics['cpu_percent_peak']:.0f}%",
                    "ram_mb": f"{resource_metrics['ram_mb_avg']:.0f}",
                }
                if "gpu_util_avg" in resource_metrics:
                    resources["gpu_util"] = f"{resource_metrics['gpu_util_avg']:.0f}%"
                if "gpu_mem_mb_avg" in resource_metrics:
                    resources["gpu_mem_mb"] = (
                        f"{resource_metrics['gpu_mem_mb_avg']:.0f}"
                    )
                if "gpu_temp_avg" in resource_metrics:
                    resources["gpu_temp"] = f"{resource_metrics['gpu_temp_avg']:.0f}"
                if "gpu_power_avg" in resource_metrics:
                    resources["gpu_power"] = f"{resource_metrics['gpu_power_avg']:.0f}"

            logger.log_epoch_line(
                epoch=epoch_num,
                total_epochs=args.epochs,
                losses={
                    "loss": avg_loss,
                    "pol": avg_policy_loss,
                    "val": avg_value_loss,
                },
                accuracy={"top1": top1 * 100, "top5": top5 * 100},
                lr=current_lr,
                grad_norms=step_grad_norms,
                samples_per_sec=samples_per_sec,
                duration_s=epoch_duration,
                resources=resources,
                data_pct=data_time / max(epoch_duration, 1e-9) * 100,
                overflows=overflow_count,
                phase="phase1",
            )
            amp_scaler_scale = trainer.amp_scaler_scale()
            scaler_state = "off"
            if amp_scaler_scale is not None:
                scaler_state = f"{amp_scaler_scale:.0f}"
            log.info(
                (
                    "Epoch %d/%d summary: loss=%.4f pol=%.4f val=%.4f "
                    "top1=%.2f%% top5=%.2f%% selected=%s(%.2f%%) "
                    "lr=%.2e sps=%.0f data=%.1f%% ovf=%d(%.3f%%) amp=%s scaler=%s"
                ),
                epoch_num,
                args.epochs,
                avg_loss,
                avg_policy_loss,
                avg_value_loss,
                top1 * 100,
                top5 * 100,
                selected_source,
                selected_top1 * 100,
                current_lr,
                samples_per_sec,
                data_time / max(epoch_duration, 1e-9) * 100.0,
                overflow_count,
                overflow_frac * 100.0,
                _amp_dtype_name(trainer.amp_dtype),
                scaler_state,
            )

            # --- Grokking detection: evaluate all holdout splits ---
            if grok_tracker is not None:
                holdout_results: dict[str, tuple[float, float]] = {
                    "random": (top1, avg_loss)
                }
                eval_nonrandom_grok = boundary_epoch or (
                    epoch_num % tcfg.phase1_grok_eval_every == 0
                )
                if eval_nonrandom_grok:
                    for split_name in _GROK_HOLDOUT_SPLITS:
                        if split_name == "random":
                            continue
                        split_count = data_plan.holdout_split_counts.get(split_name, 0)
                        if split_count <= 0:
                            continue
                        split_top1, _ = _measure_accuracy_from_indices(
                            trainer=trainer,
                            data_plan=data_plan,
                            indices_by_shard=data_plan.holdout_split_indices[
                                split_name
                            ],
                            device=device,
                            use_tqdm=use_tqdm,
                            progress_desc=f"Eval {split_name} E{epoch_num}",
                        )
                        holdout_results[split_name] = (split_top1, avg_loss)
                grok_epoch_metrics = grok_tracker.epoch(
                    epoch, avg_loss, holdout_results
                )
                logger.log_grok_metrics(epoch, grok_epoch_metrics)

            # Plateau detection
            grad_norm_avg = (
                sum(step_grad_norms) / len(step_grad_norms) if step_grad_norms else 0.0
            )
            grad_norm_peak = max(step_grad_norms) if step_grad_norms else float("nan")
            plateau_warnings = plateau_detector.update(
                epoch, grad_norm_avg, avg_loss, current_lr
            )
            if any(_is_loss_stall_warning(msg) for msg in plateau_warnings):
                grokfast_plateau_streak += 1
            else:
                grokfast_plateau_streak = 0

            guard_decision = _evaluate_stability_guard(
                enabled=_STABILITY_GUARD_ENABLED,
                epoch_num=epoch_num,
                best_top1=best_acc,
                current_top1=selected_top1,
                overflow_frac=overflow_frac,
                grad_peak=grad_norm_peak,
                backoff_count=stability_backoff_count,
                cooldown_until_epoch=stability_guard_cooldown_until_epoch,
                min_epoch=_STABILITY_GUARD_MIN_EPOCH,
                drop_ratio=_STABILITY_GUARD_DROP_RATIO,
                min_drop=_STABILITY_GUARD_MIN_DROP,
                overflow_frac_threshold=_STABILITY_GUARD_OVERFLOW_FRAC,
                grad_peak_threshold=_STABILITY_GUARD_GRAD_PEAK,
                max_backoffs=_STABILITY_GUARD_MAX_BACKOFFS,
            )
            if guard_decision.trigger:
                rollback_path = Path(args.output)
                log.warning(
                    "Stability guard triggered at epoch %d: %s",
                    epoch_num,
                    guard_decision.reason,
                )
                restored = _restore_phase1_from_checkpoint(
                    checkpoint_path=rollback_path,
                    device=device,
                    trainer=trainer,
                    model_ema=model_ema,
                )
                if not restored:
                    log.error(
                        "Stability guard could not restore best checkpoint at %s; stopping.",
                        rollback_path,
                    )
                    break
                old_lrs, new_lrs = trainer.backoff_learning_rates(
                    _STABILITY_GUARD_LR_BACKOFF,
                    min_lr=tcfg.min_lr,
                )
                trainer.reset_amp_scaler()
                trainer.reset_grokfast_filter()
                stability_backoff_count += 1
                stability_guard_cooldown_until_epoch = (
                    epoch_num + _STABILITY_GUARD_COOLDOWN_EPOCHS
                )
                consecutive_overflow_epochs = 0

                if (
                    not grokfast_disabled_by_guard
                    and tcfg.grokfast
                    and stability_backoff_count >= _STABILITY_GUARD_DISABLE_GROKFAST_AFTER
                ):
                    disabled_active_filter = trainer.disable_grokfast_filter()
                    if disabled_active_filter:
                        grokfast_enabled = False
                        grokfast_disabled_by_guard = True
                        log.warning(
                            "Stability guard disabled Grokfast after %d rollback(s).",
                            stability_backoff_count,
                        )
                    elif not grokfast_enabled:
                        grokfast_disabled_by_guard = True
                        log.warning(
                            (
                                "Stability guard suppressed delayed Grokfast activation "
                                "after %d rollback(s)."
                            ),
                            stability_backoff_count,
                        )

                old_head_lr = max(old_lrs)
                new_head_lr = max(new_lrs)
                log.warning(
                    (
                        "Stability guard rollback complete: restored=%s "
                        "head_lr %.2e->%.2e backoff=%d/%d cooldown_until_epoch=%d"
                    ),
                    rollback_path,
                    old_head_lr,
                    new_head_lr,
                    stability_backoff_count,
                    _STABILITY_GUARD_MAX_BACKOFFS,
                    stability_guard_cooldown_until_epoch,
                )
                continue

            if (
                overflow_backoff_streak >= _OVERFLOW_BACKOFF_STREAK
                and epoch_num >= _OVERFLOW_BACKOFF_MIN_EPOCH
                and overflow_backoff_count < _OVERFLOW_BACKOFF_MAX_BACKOFFS
                and epoch_num > overflow_backoff_cooldown_until_epoch
            ):
                old_lrs, new_lrs = trainer.backoff_learning_rates(
                    _OVERFLOW_BACKOFF_LR_FACTOR,
                    min_lr=tcfg.min_lr,
                )
                trainer.reset_amp_scaler()
                if grokfast_enabled:
                    trainer.reset_grokfast_filter()
                overflow_backoff_count += 1
                overflow_backoff_streak = 0
                overflow_backoff_cooldown_until_epoch = (
                    epoch_num + _OVERFLOW_BACKOFF_COOLDOWN_EPOCHS
                )
                log.warning(
                    (
                        "Overflow mitigation applied: overflow_frac=%.3f%% "
                        "for %d consecutive epoch(s). head_lr %.2e->%.2e "
                        "backoff=%d/%d cooldown_until_epoch=%d"
                    ),
                    overflow_frac * 100.0,
                    _OVERFLOW_BACKOFF_STREAK,
                    max(old_lrs),
                    max(new_lrs),
                    overflow_backoff_count,
                    _OVERFLOW_BACKOFF_MAX_BACKOFFS,
                    overflow_backoff_cooldown_until_epoch,
                )

            if selected_top1 > best_acc:
                best_acc = selected_top1
                best_source = selected_source
                if selected_source == "swa":
                    with swa_model.apply():
                        _save_current_checkpoint()
                elif selected_source == "ema" and model_ema is not None:
                    with model_ema.apply():
                        _save_current_checkpoint()
                else:
                    _save_current_checkpoint()

            # Phase gate check
            if selected_top1 > tcfg.phase1_gate:
                log.info(
                    "PHASE 1 GATE PASSED: top-1 accuracy %s (%s) > %s — ready for Phase 2",
                    f"{selected_top1:.1%}",
                    selected_source,
                    f"{tcfg.phase1_gate:.0%}",
                )
                break
            if stop_for_overflow:
                break

    if grok_tracker is not None:
        grok_tracker.close()

    log.info("best_top1=%s source=%s", f"{best_acc:.1%}", best_source)


if __name__ == "__main__":
    main()
