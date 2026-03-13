"""Execution-domain training utilities."""

from denoisr_crypto.training.baseline import (
    BaselineArtifacts,
    train_baseline,
)
from denoisr_crypto.training.entry_quality import (
    EntryTrainingArtifacts,
    train_entry_quality_model,
)
from denoisr_crypto.training.losses import (
    LossArtifacts,
    LossConfig,
    compute_entry_loss,
    temperature_for_epoch,
)

__all__ = [
    "BaselineArtifacts",
    "EntryTrainingArtifacts",
    "LossArtifacts",
    "LossConfig",
    "compute_entry_loss",
    "temperature_for_epoch",
    "train_baseline",
    "train_entry_quality_model",
]
