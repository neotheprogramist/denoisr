import chess
import pytest
import torch
from typing import Any

from denoisr_chess.training.mcts import MCTS, MCTSConfig

from conftest import SMALL_D_S


class _MockPolicyValue:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy = torch.ones(64, 64) / (64 * 64)
        value = torch.tensor([0.33, 0.34, 0.33])
        return policy, value


class _MockWorldModel:
    def __init__(self, d_s: int) -> None:
        self.d_s = d_s

    def predict_next(
        self, state: torch.Tensor, f: int, t: int
    ) -> tuple[torch.Tensor, float]:
        return torch.randn(64, self.d_s), 0.0


def _fixed_legal_mask_fn(board: Any) -> torch.Tensor:
    """Return the board object itself, which is a bool mask tensor."""
    assert isinstance(board, torch.Tensor)
    return board


def _identity_transition_fn(board: Any, f: int, t: int) -> Any:
    """Transition that preserves the same legal mask as the board."""
    return board


def _make_mcts(
    legal_mask: torch.Tensor,
    config: MCTSConfig,
    policy_value_fn: Any | None = None,
) -> MCTS:
    """Build an MCTS instance using the legal_mask tensor as a mock board."""
    pv = _MockPolicyValue(SMALL_D_S) if policy_value_fn is None else None
    wm = _MockWorldModel(SMALL_D_S)
    return MCTS(
        policy_value_fn=policy_value_fn if policy_value_fn is not None else pv.predict,
        world_model_fn=wm.predict_next,
        config=config,
        legal_mask_fn=_fixed_legal_mask_fn,
        transition_fn=_identity_transition_fn,
    )


class TestMCTS:
    @pytest.fixture
    def config(self) -> MCTSConfig:
        return MCTSConfig(
            num_simulations=50,
            c_puct=1.4,
            dirichlet_alpha=0.3,
            dirichlet_epsilon=0.25,
        )

    def test_search_returns_valid_distribution(self, config: MCTSConfig) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True
        legal_mask[1, 18] = True

        mcts = _make_mcts(legal_mask, config)
        visit_dist = mcts.search(state, legal_mask, root_board=legal_mask)
        assert visit_dist.shape == (64, 64)
        assert visit_dist.sum().item() == pytest.approx(1.0, abs=1e-5)

    def test_search_only_legal_moves(self, config: MCTSConfig) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        mcts = _make_mcts(legal_mask, config)
        visit_dist = mcts.search(state, legal_mask, root_board=legal_mask)
        illegal_mass = visit_dist[~legal_mask].sum().item()
        assert illegal_mass == pytest.approx(0.0, abs=1e-7)

    def test_more_sims_concentrates_distribution(self) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        config_few = MCTSConfig(num_simulations=10, c_puct=1.4)
        config_many = MCTSConfig(num_simulations=200, c_puct=1.4)
        mcts_few = _make_mcts(legal_mask, config_few)
        mcts_many = _make_mcts(legal_mask, config_many)

        dist_few = mcts_few.search(state, legal_mask, root_board=legal_mask)
        dist_many = mcts_many.search(state, legal_mask, root_board=legal_mask)

        entropy_few = -(dist_few[dist_few > 0] * dist_few[dist_few > 0].log()).sum()
        entropy_many = -(
            dist_many[dist_many > 0] * dist_many[dist_many > 0].log()
        ).sum()
        assert entropy_many <= entropy_few + 0.5

    def test_visit_counts_nonnegative(self, config: MCTSConfig) -> None:
        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True

        mcts = _make_mcts(legal_mask, config)
        visit_dist = mcts.search(state, legal_mask, root_board=legal_mask)
        assert (visit_dist >= 0).all()

    def test_search_accepts_logit_policy(self) -> None:
        class _LogitPV:
            def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                logits = torch.full((64, 64), -5.0)
                logits[12, 28] = 5.0
                logits[12, 20] = 1.0
                value = torch.tensor([0.6, 0.2, 0.2])
                return logits, value

        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        mcts = _make_mcts(
            legal_mask,
            MCTSConfig(num_simulations=50, c_puct=1.4),
            policy_value_fn=_LogitPV().predict,
        )
        state = torch.randn(64, SMALL_D_S)
        dist = mcts.search(state, legal_mask, root_board=legal_mask)
        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
        assert dist[12, 28].item() > dist[12, 20].item()

    def test_temperature_controls_visit_distribution_sharpness(self) -> None:
        class _LogitPV:
            def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                logits = torch.full((64, 64), -5.0)
                logits[12, 28] = 5.0
                logits[12, 20] = 1.0
                value = torch.tensor([0.6, 0.2, 0.2])
                return logits, value

        state = torch.randn(64, SMALL_D_S)
        legal_mask = torch.zeros(64, 64, dtype=torch.bool)
        legal_mask[12, 28] = True
        legal_mask[12, 20] = True

        low_temp = _make_mcts(
            legal_mask,
            MCTSConfig(
                num_simulations=80,
                c_puct=1.4,
                dirichlet_epsilon=0.0,
                temperature=0.25,
            ),
            policy_value_fn=_LogitPV().predict,
        )
        high_temp = _make_mcts(
            legal_mask,
            MCTSConfig(
                num_simulations=80,
                c_puct=1.4,
                dirichlet_epsilon=0.0,
                temperature=2.0,
            ),
            policy_value_fn=_LogitPV().predict,
        )

        dist_low = low_temp.search(state, legal_mask, root_board=legal_mask)
        dist_high = high_temp.search(state, legal_mask, root_board=legal_mask)
        assert dist_low[12, 28].item() > dist_high[12, 28].item()

    def test_board_aware_legality_applies_beyond_root(self) -> None:
        class _UniformPV:
            def predict(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.zeros(64, 64), torch.tensor([0.4, 0.2, 0.4])

        wm = _MockWorldModel(SMALL_D_S)
        board = chess.Board()

        def legal_mask_fn(b: chess.Board) -> torch.Tensor:
            mask = torch.zeros(64, 64, dtype=torch.bool)
            for move in b.legal_moves:
                mask[move.from_square, move.to_square] = True
            return mask

        def transition_fn(b: chess.Board, f: int, t: int) -> chess.Board:
            move = chess.Move(f, t)
            assert move in b.legal_moves
            next_board = b.copy()
            next_board.push(move)
            return next_board

        mcts = MCTS(
            policy_value_fn=_UniformPV().predict,
            world_model_fn=wm.predict_next,
            config=MCTSConfig(num_simulations=80, c_puct=1.4),
            legal_mask_fn=legal_mask_fn,
            transition_fn=transition_fn,
        )
        state = torch.randn(64, SMALL_D_S)
        root_legal = legal_mask_fn(board)
        dist = mcts.search(state, root_legal, root_board=board)
        assert dist.sum().item() == pytest.approx(1.0, abs=1e-5)
