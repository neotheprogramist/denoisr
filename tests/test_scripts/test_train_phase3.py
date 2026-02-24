from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import denoisr.scripts.train_phase3 as train_phase3
from denoisr.types import Action, GameRecord, PolicyTarget, TrainingExample, ValueTarget
from denoisr.types.board import BoardTensor


class _DummyTqdm:
    def __init__(
        self,
        iterable: Any | None = None,
        *,
        total: int | None = None,
        **_: object,
    ) -> None:
        if iterable is None:
            self._iterable = range(total or 0)
        else:
            self._iterable = iterable

    def __iter__(self) -> Any:
        return iter(self._iterable)

    def update(self, _n: int = 1) -> None:
        return None

    def set_postfix(self, **_: object) -> None:
        return None

    def close(self) -> None:
        return None


class _DummyEncoder(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"expected 4D input, got {x.ndim}D")
        b = x.shape[0]
        return torch.zeros(b, 64, 4, device=x.device)


class _DummyBackbone(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(f"expected 3D input, got {x.ndim}D")
        return x


class _DummyPolicyHead(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        return torch.zeros(b, 64, 64, device=x.device)


class _DummyValueHead(nn.Module):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b = x.shape[0]
        return torch.zeros(b, 3, device=x.device), torch.zeros(b, 1, device=x.device)


class _DummyWorldModel(nn.Module):
    def forward(
        self,
        states: torch.Tensor,
        action_from: torch.Tensor,
        action_to: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, _, d = states.shape
        _ = action_from, action_to
        next_states = torch.zeros(b, t, 64, d, device=states.device)
        rewards = torch.zeros(b, t, device=states.device)
        return next_states, rewards


class _DummyDiffusion(nn.Module):
    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor,
    ) -> torch.Tensor:
        _ = t, cond
        return torch.zeros_like(x)

    def fuse(self, latent: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
        _ = denoised
        return latent


class _DummyConsistency(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mean(dim=1)


class _DummySchedule:
    num_timesteps = 8

    def to(self, _device: torch.device) -> "_DummySchedule":
        return self


class _DummyBoardEncoder:
    def encode(self, _board: object) -> BoardTensor:
        return BoardTensor(torch.zeros(122, 8, 8, dtype=torch.float32))


class _FakeSolver:
    def __init__(self, schedule: object, num_steps: int = 10) -> None:
        self.schedule = schedule
        self.num_steps = num_steps

    def sample(
        self,
        model_fn: object,
        shape: tuple[int, ...],
        cond: torch.Tensor,
        device: torch.device,
    ) -> torch.Tensor:
        _ = model_fn, cond
        return torch.zeros(shape, device=device)


def test_records_to_trajectory_batch_window_and_rewards() -> None:
    record = GameRecord(
        actions=(
            Action(12, 28),  # white move
            Action(52, 36),  # black move
        ),
        result=1.0,
    )
    batch = train_phase3._records_to_trajectory_batch(
        [record],
        board_encoder=_DummyBoardEncoder(),
        seq_len=2,
    )
    assert batch is not None
    assert batch.boards.shape == (1, 2, 122, 8, 8)
    assert batch.actions_from.shape == (1, 1)
    assert batch.actions_to.shape == (1, 1)
    assert batch.policies.shape == (1, 1, 64, 64)
    assert batch.rewards.shape == (1, 1)
    assert batch.rewards[0, 0].item() == pytest.approx(1.0)


def test_records_to_trajectory_batch_returns_none_when_too_short() -> None:
    short = GameRecord(actions=(Action(12, 28),), result=1.0)
    batch = train_phase3._records_to_trajectory_batch(
        [short],
        board_encoder=_DummyBoardEncoder(),
        seq_len=3,
    )
    assert batch is None


def test_main_phase3_alpha_mixing_and_aux_updates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeSelfPlayActor:
        alpha_calls: list[float] = []
        encode_checked = False

        def __init__(
            self,
            *,
            encode_fn: object,
            diffusion_policy_fn: object,
            **_: object,
        ) -> None:
            self._encode_fn = encode_fn
            assert diffusion_policy_fn is not None

        def play_game(self, generation: int = 0, alpha: float = 0.0) -> GameRecord:
            _ = generation
            _FakeSelfPlayActor.alpha_calls.append(alpha)
            encoded = self._encode_fn(torch.zeros(1, 122, 8, 8))
            assert encoded.shape == (64, 4)
            _FakeSelfPlayActor.encode_checked = True
            return GameRecord(actions=(Action(12, 28),), result=1.0)

    class _FakeReanalyseActor:
        alpha_calls: list[float] = []

        def __init__(self, **_: object) -> None:
            return None

        def reanalyse(
            self, record: GameRecord, alpha: float = 0.0
        ) -> list[TrainingExample]:
            _ = record
            _FakeReanalyseActor.alpha_calls.append(alpha)
            policy = torch.zeros(64, 64, dtype=torch.float32)
            policy[12, 28] = 1.0
            return [
                TrainingExample(
                    board=BoardTensor(torch.zeros(122, 8, 8, dtype=torch.float32)),
                    policy=PolicyTarget(policy),
                    value=ValueTarget(win=1.0, draw=0.0, loss=0.0),
                )
            ]

    class _FakeBuffer:
        def __init__(self, capacity: int) -> None:
            _ = capacity
            self._records: list[GameRecord] = []

        def add(self, record: GameRecord, priority: float = 1.0) -> None:
            _ = priority
            self._records.append(record)

        def __len__(self) -> int:
            return len(self._records)

        def sample(self, n: int) -> list[GameRecord]:
            return self._records[-n:]

    class _FakeSupervisedTrainer:
        instances: list["_FakeSupervisedTrainer"] = []

        def __init__(self, **_: object) -> None:
            self.train_calls = 0
            self.scheduler_calls = 0
            self.optimizer = SimpleNamespace(state_dict=lambda: {"sup": 1})
            _FakeSupervisedTrainer.instances.append(self)

        def train_step(
            self, batch: list[TrainingExample]
        ) -> tuple[float, dict[str, float]]:
            _ = batch
            self.train_calls += 1
            return 0.5, {}

        def scheduler_step(self) -> None:
            self.scheduler_calls += 1

    class _FakePhase2Trainer:
        instances: list["_FakePhase2Trainer"] = []

        def __init__(self, **_: object) -> None:
            self.train_calls = 0
            self.advance_calls = 0
            self.optimizer = SimpleNamespace(state_dict=lambda: {"aux": 1})
            _FakePhase2Trainer.instances.append(self)

        def train_step(
            self, batch: train_phase3.TrajectoryBatch
        ) -> tuple[float, dict[str, float]]:
            _ = batch
            self.train_calls += 1
            return 0.25, {}

        def advance_curriculum(self) -> None:
            self.advance_calls += 1

    saved: list[dict[str, object]] = []

    def _fake_save_checkpoint(path: object, cfg: object, **kwargs: object) -> None:
        _ = path, cfg
        saved.append(kwargs)

    tcfg = SimpleNamespace(
        workers=0,
        temperature_base=1.0,
        temperature_explore_moves=30,
        temperature_generation_decay=0.97,
        max_moves=8,
        c_puct=1.4,
        dirichlet_alpha=0.3,
        dirichlet_epsilon=0.25,
        reanalyse_simulations=4,
        phase1_gate=0.5,
        phase2_gate=5.0,
        policy_weight=1.0,
        value_weight=1.0,
        illegal_penalty_weight=0.0,
        use_harmony_dream=False,
        harmony_ema_decay=0.99,
        warmup_epochs=1,
        max_grad_norm=1.0,
        weight_decay=0.0,
        encoder_lr_multiplier=1.0,
        min_lr=1e-6,
        use_warm_restarts=False,
        consistency_weight=1.0,
        diffusion_weight=1.0,
        reward_weight=1.0,
        ply_weight=0.1,
        curriculum_initial_fraction=0.25,
        curriculum_growth=1.02,
    )

    cfg = SimpleNamespace(d_s=4)
    state = {
        "encoder": {},
        "backbone": {},
        "policy_head": {},
        "value_head": {},
        "world_model": {},
        "diffusion": {},
        "consistency": {},
    }

    monkeypatch.setattr(train_phase3, "tqdm", _DummyTqdm)
    monkeypatch.setattr(train_phase3, "detect_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(train_phase3, "resolve_dataloader_workers", lambda _w: 1)
    monkeypatch.setattr(train_phase3, "full_training_config_from_args", lambda _a: tcfg)
    monkeypatch.setattr(train_phase3, "load_checkpoint", lambda _p, _d: (cfg, state))
    monkeypatch.setattr(train_phase3, "build_encoder", lambda _cfg: _DummyEncoder())
    monkeypatch.setattr(train_phase3, "build_backbone", lambda _cfg: _DummyBackbone())
    monkeypatch.setattr(
        train_phase3, "build_policy_head", lambda _cfg: _DummyPolicyHead()
    )
    monkeypatch.setattr(
        train_phase3, "build_value_head", lambda _cfg: _DummyValueHead()
    )
    monkeypatch.setattr(
        train_phase3, "build_world_model", lambda _cfg: _DummyWorldModel()
    )
    monkeypatch.setattr(train_phase3, "build_diffusion", lambda _cfg: _DummyDiffusion())
    monkeypatch.setattr(
        train_phase3, "build_consistency", lambda _cfg: _DummyConsistency()
    )
    monkeypatch.setattr(train_phase3, "build_schedule", lambda _cfg: _DummySchedule())
    monkeypatch.setattr(
        train_phase3, "build_board_encoder", lambda _cfg: _DummyBoardEncoder()
    )
    monkeypatch.setattr(train_phase3, "DPMSolverPP", _FakeSolver)
    monkeypatch.setattr(train_phase3, "SelfPlayActor", _FakeSelfPlayActor)
    monkeypatch.setattr(train_phase3, "ReanalyseActor", _FakeReanalyseActor)
    monkeypatch.setattr(train_phase3, "PriorityReplayBuffer", _FakeBuffer)
    monkeypatch.setattr(train_phase3, "SupervisedTrainer", _FakeSupervisedTrainer)
    monkeypatch.setattr(train_phase3, "Phase2Trainer", _FakePhase2Trainer)
    monkeypatch.setattr(train_phase3, "save_checkpoint", _fake_save_checkpoint)

    out = tmp_path / "phase3.pt"
    monkeypatch.setattr(
        "sys.argv",
        [
            "denoisr-train-phase3",
            "--checkpoint",
            "fake_phase2.pt",
            "--generations",
            "2",
            "--games-per-gen",
            "1",
            "--reanalyse-per-gen",
            "1",
            "--alpha-generations",
            "1",
            "--train-batch-size",
            "1",
            "--aux-updates-per-gen",
            "2",
            "--aux-batch-size",
            "1",
            "--aux-seq-len",
            "2",
            "--save-every",
            "1",
            "--output",
            str(out),
        ],
    )

    train_phase3.main()

    assert _FakeSelfPlayActor.encode_checked is True
    assert _FakeSelfPlayActor.alpha_calls == [0.0, 1.0]
    assert _FakeReanalyseActor.alpha_calls == [0.0, 1.0]

    assert len(_FakePhase2Trainer.instances) == 1
    aux = _FakePhase2Trainer.instances[0]
    assert aux.train_calls == 4  # 2 generations * 2 aux updates/gen
    assert aux.advance_calls == 2

    assert len(_FakeSupervisedTrainer.instances) == 1
    sup = _FakeSupervisedTrainer.instances[0]
    assert sup.train_calls == 2
    assert sup.scheduler_calls == 2

    assert len(saved) == 2
    assert all("phase2_optimizer" in call for call in saved)
