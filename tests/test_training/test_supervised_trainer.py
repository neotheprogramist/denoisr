import pathlib
import math

import pytest
import torch

from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.training.grokfast import GrokfastFilter
from denoisr.training.loss import ChessLossComputer
from denoisr.training.supervised_trainer import SupervisedTrainer
from denoisr.types import BoardTensor, PolicyTarget, TrainingExample, ValueTarget

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
)


def _make_batch(n: int = 8) -> list[TrainingExample]:
    examples = []
    for _ in range(n):
        board = BoardTensor(torch.randn(12, 8, 8))
        policy_data = torch.zeros(64, 64)
        policy_data[12, 28] = 1.0
        policy = PolicyTarget(policy_data)
        value = ValueTarget(win=1.0, draw=0.0, loss=0.0)
        examples.append(TrainingExample(board=board, policy=policy, value=value))
    return examples


class TestSupervisedTrainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> SupervisedTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        return SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "policy" in breakdown

    def test_loss_decreases_over_steps(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        losses = []
        for _ in range(20):
            loss, _ = trainer.train_step(batch)
            losses.append(loss)
        assert losses[-1] < losses[0]

    def test_save_and_load_checkpoint(
        self, trainer: SupervisedTrainer, tmp_path: pathlib.Path
    ) -> None:
        path = tmp_path / "checkpoint.pt"
        trainer.save_checkpoint(path)
        assert path.exists()
        trainer.load_checkpoint(path)

    def test_encoder_lr_lower_than_head_lr(self, trainer: SupervisedTrainer) -> None:
        groups = trainer.optimizer.param_groups
        encoder_lr = groups[0]["lr"]
        head_lr = groups[2]["lr"]
        assert encoder_lr < head_lr

    def test_gradients_are_clipped(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        trainer.train_step(batch)
        all_params = [
            p
            for group in trainer.optimizer.param_groups
            for p in group["params"]
            if p.grad is not None
        ]
        total_norm = torch.nn.utils.clip_grad_norm_(all_params, float("inf"))
        assert total_norm.item() < 100.0

    def test_breakdown_includes_batch_top1(self, trainer: SupervisedTrainer) -> None:
        batch = _make_batch(4)
        _, breakdown = trainer.train_step(batch)
        assert "batch_top1" in breakdown
        assert 0.0 <= breakdown["batch_top1"] <= 1.0

    def test_lr_reaches_min_at_final_epoch(self, device: torch.device) -> None:
        """Cosine schedule should reach eta_min at the final epoch."""
        total_epochs = 100
        warmup = 5
        min_lr = 1e-6
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=3e-4,
            device=device,
            total_epochs=total_epochs,
            warmup_epochs=warmup,
            min_lr=min_lr,
        )

        # Run through all epochs
        for _ in range(total_epochs):
            trainer.scheduler_step()

        head_lr = trainer.optimizer.param_groups[2]["lr"]
        # LR should be at or near eta_min at the end
        assert head_lr < min_lr * 2

    def test_warm_restarts_produce_lr_spikes(self, device: torch.device) -> None:
        """Warm restarts should produce periodic LR resets."""
        total_epochs = 60
        warmup = 3
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=3e-4,
            device=device,
            total_epochs=total_epochs,
            warmup_epochs=warmup,
            use_warm_restarts=True,
        )

        lrs: list[float] = []
        for _ in range(total_epochs):
            trainer.scheduler_step()
            lrs.append(trainer.optimizer.param_groups[2]["lr"])

        # With warm restarts, LR should increase at some point after warmup
        post_warmup_lrs = lrs[warmup:]
        has_increase = any(
            post_warmup_lrs[i + 1] > post_warmup_lrs[i]
            for i in range(len(post_warmup_lrs) - 1)
        )
        assert has_increase, (
            "Warm restarts should cause LR to increase at restart points"
        )

    def test_scheduler_reduces_lr(self, trainer: SupervisedTrainer) -> None:
        """After warmup + cosine decay, LRs should be below peak."""
        peak_lrs = trainer._base_lrs
        batch = _make_batch(8)
        # Run through warmup + 5 cosine decay steps
        for _ in range(trainer._warmup_epochs + 5):
            trainer.train_step(batch)
            trainer.scheduler_step()
        current_lrs = [g["lr"] for g in trainer.optimizer.param_groups]
        # After cosine decay, LRs should be below peak
        assert all(c < p for c, p in zip(current_lrs, peak_lrs))

    def test_marks_overflow_on_nonfinite_loss(
        self, trainer: SupervisedTrainer, device: torch.device
    ) -> None:
        boards = torch.randn(4, 12, 8, 8, device=device)
        policies = torch.zeros(4, 64, 64, device=device)
        policies[:, 12, 28] = 1.0
        values = torch.zeros(4, 3, device=device)
        values[:, 1] = 1.0
        values[0, 0] = float("nan")
        values[0, 1] = 0.0

        loss, breakdown = trainer.train_step_tensors(boards, policies, values)
        assert math.isnan(loss)
        assert breakdown["overflow"] is True
        assert math.isnan(float(breakdown["grad_norm"]))

    def test_backoff_learning_rates_scales_optimizer_and_scheduler(
        self, trainer: SupervisedTrainer
    ) -> None:
        old_current = [float(group["lr"]) for group in trainer.optimizer.param_groups]
        old_base = list(trainer._base_lrs)
        old_scheduler_base = list(trainer._scheduler.base_lrs)

        old_lrs, new_lrs = trainer.backoff_learning_rates(0.5, min_lr=1e-8)

        assert old_lrs == old_current
        assert new_lrs == [pytest.approx(v * 0.5) for v in old_current]
        assert trainer._base_lrs == [pytest.approx(v * 0.5) for v in old_base]
        assert trainer._scheduler.base_lrs == [
            pytest.approx(v * 0.5) for v in old_scheduler_base
        ]

    def test_backoff_learning_rates_rejects_nonpositive_factor(
        self, trainer: SupervisedTrainer
    ) -> None:
        with pytest.raises(ValueError, match="factor"):
            trainer.backoff_learning_rates(0.0)

    def test_set_and_disable_grokfast_filter(self, trainer: SupervisedTrainer) -> None:
        gf = GrokfastFilter(alpha=0.98, lamb=1.0)
        trainer.set_grokfast_filter(gf)
        assert trainer.disable_grokfast_filter() is True
        assert trainer.disable_grokfast_filter() is False


class TestSupervisedTrainerGrokfast:
    @pytest.fixture
    def trainer_with_grokfast(self, device: torch.device) -> SupervisedTrainer:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        loss_fn = ChessLossComputer()
        gf = GrokfastFilter(alpha=0.98, lamb=2.0)
        return SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
            grokfast_filter=gf,
        )

    def test_train_step_with_grokfast(
        self, trainer_with_grokfast: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        loss, breakdown = trainer_with_grokfast.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_grokfast_ema_populated_after_step(
        self, trainer_with_grokfast: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        trainer_with_grokfast.train_step(batch)
        assert trainer_with_grokfast._grokfast_filter is not None
        assert len(trainer_with_grokfast._grokfast_filter.grads) > 0

    def test_recovers_after_nonfinite_batch_with_grokfast(
        self, trainer_with_grokfast: SupervisedTrainer, device: torch.device
    ) -> None:
        boards = torch.randn(4, 12, 8, 8, device=device)
        policies = torch.zeros(4, 64, 64, device=device)
        policies[:, 12, 28] = 1.0
        values = torch.zeros(4, 3, device=device)
        values[:, 1] = 1.0
        values[0, 0] = float("nan")
        values[0, 1] = 0.0

        _, bad_breakdown = trainer_with_grokfast.train_step_tensors(
            boards, policies, values
        )
        assert bad_breakdown["overflow"] is True

        clean_values = torch.zeros(4, 3, device=device)
        clean_values[:, 1] = 1.0
        clean_loss, clean_breakdown = trainer_with_grokfast.train_step_tensors(
            boards, policies, clean_values
        )
        assert math.isfinite(clean_loss)
        assert clean_breakdown["overflow"] is False

    def test_can_reset_and_disable_grokfast_filter(
        self, trainer_with_grokfast: SupervisedTrainer
    ) -> None:
        batch = _make_batch(4)
        trainer_with_grokfast.train_step(batch)
        assert trainer_with_grokfast._grokfast_filter is not None
        assert len(trainer_with_grokfast._grokfast_filter.grads) > 0

        trainer_with_grokfast.reset_grokfast_filter()
        assert trainer_with_grokfast._grokfast_filter is not None
        assert trainer_with_grokfast._grokfast_filter.grads == {}

        assert trainer_with_grokfast.disable_grokfast_filter() is True
        assert trainer_with_grokfast._grokfast_filter is None
        assert trainer_with_grokfast.disable_grokfast_filter() is False


class TestOneCycleLR:
    def test_onecycle_scheduler_accepted(self) -> None:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(
            SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
        )
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        loss_fn = ChessLossComputer()

        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            use_onecycle=True,
            steps_per_epoch=100,
            total_epochs=10,
        )
        assert isinstance(trainer._scheduler, torch.optim.lr_scheduler.OneCycleLR)


class TestGradientAccumulation:
    def test_accumulation_steps_accepted(self) -> None:
        encoder = ChessEncoder(num_planes=12, d_s=SMALL_D_S)
        backbone = ChessPolicyBackbone(
            SMALL_D_S, SMALL_NUM_HEADS, SMALL_NUM_LAYERS, SMALL_FFN_DIM
        )
        policy_head = ChessPolicyHead(d_s=SMALL_D_S)
        value_head = ChessValueHead(d_s=SMALL_D_S)
        loss_fn = ChessLossComputer()

        trainer = SupervisedTrainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            loss_fn=loss_fn,
            lr=1e-3,
            accumulation_steps=4,
        )
        assert trainer._accum_steps == 4
