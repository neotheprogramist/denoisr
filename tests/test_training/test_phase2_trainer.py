import math
from pathlib import Path

import pytest
import torch

from conftest import (
    SMALL_D_S,
    SMALL_FFN_DIM,
    SMALL_NUM_HEADS,
    SMALL_NUM_LAYERS,
    SMALL_NUM_TIMESTEPS,
)
from denoisr.data.extended_board_encoder import ExtendedBoardEncoder
from denoisr.nn.consistency import ChessConsistencyProjector
from denoisr.nn.diffusion import ChessDiffusionModule, CosineNoiseSchedule
from denoisr.nn.encoder import ChessEncoder
from denoisr.nn.policy_backbone import ChessPolicyBackbone
from denoisr.nn.policy_head import ChessPolicyHead
from denoisr.nn.value_head import ChessValueHead
from denoisr.nn.world_model import ChessWorldModel
from denoisr.scripts.train_phase2 import extract_trajectories
from denoisr.training.loss import ChessLossComputer
from denoisr.training.phase2_trainer import (
    Phase2Trainer,
    TrajectoryBatch,
    evaluate_phase2_gate,
)


class TestTrajectoryBatch:
    def test_valid_shapes(self) -> None:
        B, T, C = 4, 5, 122
        batch = TrajectoryBatch(
            boards=torch.randn(B, T, C, 8, 8),
            actions_from=torch.randint(0, 64, (B, T - 1)),
            actions_to=torch.randint(0, 64, (B, T - 1)),
            policies=torch.zeros(B, T - 1, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * B),
            rewards=torch.zeros(B, T - 1),
        )
        assert batch.boards.shape == (B, T, C, 8, 8)
        assert batch.actions_from.shape == (B, T - 1)
        assert batch.policies.shape == (B, T - 1, 64, 64)
        assert batch.values.shape == (B, 3)
        assert batch.rewards.shape == (B, T - 1)

    def test_frozen(self) -> None:
        batch = TrajectoryBatch(
            boards=torch.randn(2, 3, 122, 8, 8),
            actions_from=torch.zeros(2, 2, dtype=torch.long),
            actions_to=torch.zeros(2, 2, dtype=torch.long),
            policies=torch.zeros(2, 2, 64, 64),
            values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
            rewards=torch.zeros(2, 2),
        )
        with pytest.raises(AttributeError):
            batch.boards = torch.randn(2, 3, 122, 8, 8)  # type: ignore[misc]

    def test_rejects_mismatched_time_dims(self) -> None:
        with pytest.raises(ValueError, match="time"):
            TrajectoryBatch(
                boards=torch.randn(2, 5, 122, 8, 8),
                actions_from=torch.zeros(2, 3, dtype=torch.long),  # should be 4
                actions_to=torch.zeros(2, 4, dtype=torch.long),
                policies=torch.zeros(2, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 2),
                rewards=torch.zeros(2, 4),
            )

    def test_rejects_mismatched_batch_dims(self) -> None:
        with pytest.raises(ValueError, match="batch"):
            TrajectoryBatch(
                boards=torch.randn(4, 5, 122, 8, 8),
                actions_from=torch.zeros(2, 4, dtype=torch.long),  # B=2 vs 4
                actions_to=torch.zeros(4, 4, dtype=torch.long),
                policies=torch.zeros(4, 4, 64, 64),
                values=torch.tensor([[1.0, 0.0, 0.0]] * 4),
                rewards=torch.zeros(4, 4),
            )


def _make_trajectory_batch(
    B: int = 2,
    T: int = 5,
    C: int = 122,
    device: torch.device = torch.device("cpu"),
) -> TrajectoryBatch:
    policies = torch.zeros(B, T - 1, 64, 64, device=device)
    for b in range(B):
        for t in range(T - 1):
            policies[b, t, 12, 28] = 1.0
    return TrajectoryBatch(
        boards=torch.randn(B, T, C, 8, 8, device=device),
        actions_from=torch.randint(0, 64, (B, T - 1), device=device),
        actions_to=torch.randint(0, 64, (B, T - 1), device=device),
        policies=policies,
        values=torch.tensor([[1.0, 0.0, 0.0]] * B, device=device),
        rewards=torch.ones(B, T - 1, device=device),
    )


class TestPhase2Trainer:
    @pytest.fixture
    def trainer(self, device: torch.device) -> Phase2Trainer:
        encoder = ChessEncoder(num_planes=122, d_s=SMALL_D_S).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        value_head = ChessValueHead(d_s=SMALL_D_S).to(device)
        world_model = ChessWorldModel(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        consistency = ChessConsistencyProjector(d_s=SMALL_D_S).to(device)
        schedule = CosineNoiseSchedule(
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        loss_fn = ChessLossComputer()
        return Phase2Trainer(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            value_head=value_head,
            world_model=world_model,
            diffusion=diffusion,
            consistency=consistency,
            schedule=schedule,
            loss_fn=loss_fn,
            lr=1e-3,
            device=device,
        )

    def test_train_step_returns_loss_and_breakdown(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0

    def test_breakdown_contains_all_6_terms(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        _, breakdown = trainer.train_step(batch)
        for key in ("policy", "value", "diffusion", "state", "reward", "consistency"):
            assert key in breakdown, f"Missing '{key}' in breakdown"
        for key in ("top1", "top5"):
            assert key in breakdown, f"Missing '{key}' in breakdown"
        assert "fused_policy" in breakdown
        assert "grad_norm" in breakdown
        assert "overflow" in breakdown
        assert breakdown["overflow"] is False

    def test_fusion_gate_parameters_update(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        before = [p.detach().clone() for p in trainer.diffusion.fusion_gate.parameters()]
        trainer.train_step(batch)
        after = [p.detach() for p in trainer.diffusion.fusion_gate.parameters()]
        assert any(not torch.equal(prev, curr) for prev, curr in zip(before, after))

    def test_amp_properties(self, trainer: Phase2Trainer) -> None:
        assert trainer.amp_dtype is None
        assert trainer.amp_autocast_enabled is False
        assert trainer.amp_scaler_enabled is False
        assert trainer.amp_scaler_scale() is None

    def test_encoder_is_frozen(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        trainer.train_step(batch)
        for p in trainer.encoder.parameters():
            assert not p.requires_grad

    def test_backbone_lr_lower_than_head_lr(
        self,
        trainer: Phase2Trainer,
    ) -> None:
        groups = trainer.optimizer.param_groups
        backbone_lr = groups[0]["lr"]
        head_lr = groups[1]["lr"]
        assert backbone_lr < head_lr

    def test_loss_is_finite(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        loss, _ = trainer.train_step(batch)
        assert loss == loss  # NaN check
        assert loss < float("inf")

    def test_loss_decreases_over_steps(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        batch = _make_trajectory_batch(device=device)
        losses = [trainer.train_step(batch)[0] for _ in range(30)]
        early_avg = sum(losses[:5]) / 5
        late_avg = sum(losses[-5:]) / 5
        assert late_avg < early_avg

    def test_train_step_with_microbatching(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
    ) -> None:
        trainer._microbatch_size = 1
        batch = _make_trajectory_batch(B=3, device=device)
        loss, breakdown = trainer.train_step(batch)
        assert isinstance(loss, float)
        assert loss > 0
        assert "grad_norm" in breakdown

    def test_marks_overflow_on_nonfinite_loss(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _make_trajectory_batch(device=device)

        def _fake_forward_loss(_: TrajectoryBatch) -> tuple[torch.Tensor, dict[str, float]]:
            return torch.tensor(float("nan"), device=device), {
                "policy": 0.0,
                "value": 0.0,
                "diffusion": 0.0,
                "consistency": 0.0,
                "state": 0.0,
                "reward": 0.0,
                "top1": 0.0,
                "top5": 0.0,
            }

        monkeypatch.setattr(trainer, "_forward_loss", _fake_forward_loss)
        loss, breakdown = trainer.train_step(batch)
        assert math.isnan(loss)
        assert breakdown["overflow"] is True
        assert math.isnan(float(breakdown["grad_norm"]))

    def test_marks_overflow_on_nonfinite_gradients(
        self,
        trainer: Phase2Trainer,
        device: torch.device,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        batch = _make_trajectory_batch(device=device)

        def _fake_forward_loss(_: TrajectoryBatch) -> tuple[torch.Tensor, dict[str, float]]:
            param = next(trainer.backbone.parameters())
            total = (param.reshape(-1)[0] * 0.0) + torch.tensor(1.0, device=device)
            return total, {
                "policy": 0.0,
                "value": 0.0,
                "diffusion": 0.0,
                "consistency": 0.0,
                "state": 0.0,
                "reward": 0.0,
                "top1": 0.0,
                "top5": 0.0,
            }

        monkeypatch.setattr(trainer, "_forward_loss", _fake_forward_loss)
        monkeypatch.setattr(trainer, "_has_nonfinite_gradients", lambda: True)
        loss, breakdown = trainer.train_step(batch)
        assert math.isfinite(loss)
        assert breakdown["overflow"] is True
        assert math.isnan(float(breakdown["grad_norm"]))

    def test_curriculum_advances(self, trainer: Phase2Trainer) -> None:
        initial = trainer.current_max_steps
        for _ in range(100):
            trainer.advance_curriculum()
        assert trainer.current_max_steps > initial

    def test_scheduler_warmup_reaches_base_lr(self, trainer: Phase2Trainer) -> None:
        base_lrs = list(trainer._base_lrs)
        for _ in range(trainer._warmup_epochs):
            trainer.scheduler_step()
        current_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        for current, base in zip(current_lrs, base_lrs):
            assert current == pytest.approx(base)

    def test_scheduler_decay_has_no_restart_spikes(self, trainer: Phase2Trainer) -> None:
        for _ in range(trainer._warmup_epochs):
            trainer.scheduler_step()
        trainer.optimizer.step()
        prev_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        for _ in range(40):
            trainer.optimizer.step()
            trainer.scheduler_step()
            current_lrs = [group["lr"] for group in trainer.optimizer.param_groups]
            for current, prev in zip(current_lrs, prev_lrs):
                assert current <= prev + 1e-12
            prev_lrs = current_lrs


class TestExtractTrajectories:
    @pytest.fixture
    def pgn_file(self, tmp_path: Path) -> Path:
        pgn = tmp_path / "test.pgn"
        pgn.write_text(
            '[Event "Test"]\n'
            '[Result "1-0"]\n'
            "\n"
            "1. e2e4 e7e5 2. g1f3 b8c6 3. f1b5 a7a6 "
            "4. b5a4 g8f6 5. e1g1 f8e7 *\n\n"
        )
        return pgn

    def test_returns_trajectory_batch(self, pgn_file: Path) -> None:
        encoder = ExtendedBoardEncoder()
        batch = extract_trajectories(
            pgn_file,
            encoder,
            seq_len=3,
            max_trajectories=100,
        )
        assert isinstance(batch, TrajectoryBatch)
        assert batch.boards.shape[1] == 3
        assert batch.actions_from.shape[1] == 2
        assert batch.actions_to.shape[1] == 2
        assert batch.policies.shape[1] == 2
        assert batch.legal_masks is not None
        assert batch.legal_masks.shape[1] == 2
        assert batch.values.shape[1] == 3

    def test_policy_targets_are_one_hot(
        self,
        pgn_file: Path,
    ) -> None:
        encoder = ExtendedBoardEncoder()
        batch = extract_trajectories(
            pgn_file,
            encoder,
            seq_len=3,
            max_trajectories=100,
        )
        for i in range(batch.policies.shape[0]):
            for t in range(batch.policies.shape[1]):
                assert batch.policies[i, t].sum().item() == (pytest.approx(1.0))

    def test_values_are_valid_wdl(self, pgn_file: Path) -> None:
        encoder = ExtendedBoardEncoder()
        batch = extract_trajectories(
            pgn_file,
            encoder,
            seq_len=3,
            max_trajectories=100,
        )
        for i in range(batch.values.shape[0]):
            assert batch.values[i].sum().item() == (pytest.approx(1.0))

    def test_rewards_match_result(self, pgn_file: Path) -> None:
        encoder = ExtendedBoardEncoder()
        batch = extract_trajectories(
            pgn_file,
            encoder,
            seq_len=3,
            max_trajectories=100,
        )
        # Game result is 1-0, so all rewards are +1 or -1
        for r in batch.rewards.flatten():
            assert abs(r.item()) == pytest.approx(1.0)

    def test_extract_includes_exact_fit_window(self, tmp_path: Path) -> None:
        """If len(boards)==seq_len, extractor should still return one trajectory."""
        pgn = tmp_path / "exact_fit.pgn"
        pgn.write_text('[Event "ExactFit"]\n[Result "1-0"]\n\n1. e2e4 e7e5 *\n\n')
        encoder = ExtendedBoardEncoder()
        batch = extract_trajectories(
            pgn,
            encoder,
            seq_len=3,
            max_trajectories=10,
        )
        assert batch.boards.shape[0] == 1


class TestPhase2Gate:
    def test_returns_three_floats(self, device: torch.device) -> None:
        encoder = ChessEncoder(
            num_planes=122,
            d_s=SMALL_D_S,
        ).to(device)
        backbone = ChessPolicyBackbone(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            ffn_dim=SMALL_FFN_DIM,
        ).to(device)
        policy_head = ChessPolicyHead(d_s=SMALL_D_S).to(device)
        diffusion = ChessDiffusionModule(
            d_s=SMALL_D_S,
            num_heads=SMALL_NUM_HEADS,
            num_layers=SMALL_NUM_LAYERS,
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)
        schedule = CosineNoiseSchedule(
            num_timesteps=SMALL_NUM_TIMESTEPS,
        ).to(device)

        boards = torch.randn(8, 122, 8, 8, device=device)
        target_from = torch.randint(0, 64, (8,), device=device)
        target_to = torch.randint(0, 64, (8,), device=device)

        single_acc, diff_acc, delta = evaluate_phase2_gate(
            encoder=encoder,
            backbone=backbone,
            policy_head=policy_head,
            diffusion=diffusion,
            schedule=schedule,
            boards=boards,
            target_from=target_from,
            target_to=target_to,
            device=device,
        )
        assert isinstance(single_acc, float)
        assert isinstance(diff_acc, float)
        assert isinstance(delta, float)
        assert 0.0 <= single_acc <= 1.0
        assert 0.0 <= diff_acc <= 1.0
