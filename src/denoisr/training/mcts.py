import math
from dataclasses import dataclass
from typing import Any, Callable

import torch
from torch import Tensor


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 100
    c_puct: float = 1.4
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature: float = 1.0


class _Node:
    __slots__ = (
        "prior",
        "visit_count",
        "value_sum",
        "children",
        "state",
        "board",
        "to_play",
        "reward_from_parent",
    )

    def __init__(
        self,
        prior: float,
        state: Tensor | None = None,
        board: Any | None = None,
        to_play: int = 1,
        reward_from_parent: float = 0.0,
    ) -> None:
        self.prior = prior
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[tuple[int, int], "_Node"] = {}
        self.state = state
        self.board = board
        self.to_play = to_play
        self.reward_from_parent = reward_from_parent

    @property
    def q_value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

    def ucb_score(self, parent_visits: int, c_puct: float) -> float:
        exploration = (
            c_puct * self.prior * math.sqrt(parent_visits) / (1 + self.visit_count)
        )
        return self.q_value + exploration


PolicyValueFn = Callable[[Tensor], tuple[Tensor, Tensor]]
WorldModelFn = Callable[[Tensor, int, int], tuple[Tensor, float]]
LegalMaskFn = Callable[[Any], Tensor]
TransitionFn = Callable[[Any, int, int], Any]


class MCTS:
    """Monte Carlo Tree Search in latent space.

    Uses a policy-value function for leaf evaluation and a world model
    for state transitions. Returns a visit-count distribution over
    legal moves after num_simulations rollouts.
    """

    def __init__(
        self,
        policy_value_fn: PolicyValueFn,
        world_model_fn: WorldModelFn,
        config: MCTSConfig,
        legal_mask_fn: LegalMaskFn | None = None,
        transition_fn: TransitionFn | None = None,
    ) -> None:
        self._pv = policy_value_fn
        self._wm = world_model_fn
        self._config = config
        self._legal_mask_fn = legal_mask_fn
        self._transition_fn = transition_fn

    def search(
        self,
        root_state: Tensor,
        legal_mask: Tensor,
        root_to_play: int = 1,
        root_board: Any | None = None,
    ) -> Tensor:
        legal_mask = legal_mask.to(dtype=torch.bool, device=root_state.device)
        root = _Node(
            prior=0.0,
            state=root_state,
            board=root_board,
            to_play=root_to_play,
        )

        policy_logits, _ = self._pv(root_state)
        # Treat policy head output as logits; normalize only over legal moves.
        masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
        has_legal = legal_mask.any()
        if has_legal:
            policy = torch.softmax(masked_logits.reshape(-1), dim=0).reshape(64, 64)
        else:
            policy = torch.zeros_like(policy_logits)

        legal_indices = legal_mask.nonzero(as_tuple=False)
        if len(legal_indices) > 0 and self._config.dirichlet_epsilon > 0:
            noise = torch.distributions.Dirichlet(
                torch.full((len(legal_indices),), self._config.dirichlet_alpha)
            ).sample()
            eps = self._config.dirichlet_epsilon
            for idx, (f, t) in enumerate(legal_indices.tolist()):
                orig = policy[f, t].item()
                policy[f, t] = (1 - eps) * orig + eps * noise[idx].item()

        for f, t in legal_indices.tolist():
            root.children[(f, t)] = _Node(
                prior=policy[f, t].item(),
                to_play=-root.to_play,
            )

        for _ in range(self._config.num_simulations):
            self._simulate(root)

        visit_dist = torch.zeros(64, 64, device=legal_mask.device)
        for (f, t), child in root.children.items():
            visit_dist[f, t] = child.visit_count

        total_visits = visit_dist.sum()
        if total_visits > 0:
            if self._config.temperature == 0:
                best = visit_dist.argmax()
                visit_dist = torch.zeros(64, 64)
                visit_dist.view(-1)[best] = 1.0
            else:
                visit_dist = visit_dist / total_visits

        return visit_dist

    def _simulate(self, root: _Node) -> float:
        node = root
        path: list[_Node] = [node]
        actions: list[tuple[int, int]] = []

        while node.children and node.visit_count > 0:
            best_action = max(
                node.children.keys(),
                key=lambda a: node.children[a].ucb_score(
                    node.visit_count, self._config.c_puct
                ),
            )
            actions.append(best_action)
            node = node.children[best_action]
            path.append(node)

        if node.state is None and actions:
            parent = path[-2]
            f, t = actions[-1]
            assert parent.state is not None
            state, reward = self._wm(parent.state, f, t)
            node.state = state
            node.reward_from_parent = reward
            if (
                node.board is None
                and parent.board is not None
                and self._transition_fn is not None
            ):
                node.board = self._transition_fn(parent.board, f, t)

        if node.state is not None:
            policy_logits, value = self._pv(node.state)
            legal_indices: list[tuple[int, int]]
            if node.board is not None and self._legal_mask_fn is not None:
                legal_mask = self._legal_mask_fn(node.board).to(
                    device=policy_logits.device, dtype=torch.bool
                )
                masked_logits = policy_logits.masked_fill(~legal_mask, float("-inf"))
                if legal_mask.any():
                    policy = torch.softmax(masked_logits.reshape(-1), dim=0).reshape(
                        64, 64
                    )
                    legal_indices = [
                        (int(f), int(t))
                        for f, t in legal_mask.nonzero(as_tuple=False).tolist()
                    ]
                else:
                    policy = torch.zeros_like(policy_logits)
                    legal_indices = []
            else:
                policy = torch.softmax(policy_logits.reshape(-1), dim=0).reshape(64, 64)
                legal_indices = [
                    (fi, ti)
                    for fi in range(64)
                    for ti in range(64)
                    if policy[fi, ti].item() > 1e-8
                ]
            # Convert white-centric value into side-to-move perspective.
            leaf_value = node.to_play * (value[0] - value[2]).item()

            if not node.children:
                for fi, ti in legal_indices:
                    p = policy[fi, ti].item()
                    if p > 1e-8:
                        node.children[(fi, ti)] = _Node(
                            prior=p,
                            to_play=-node.to_play,
                        )
        else:
            leaf_value = 0.0

        # Backup with alternating turns and edge rewards:
        # V_parent = r(parent->child) - V_child
        value_to_backup = leaf_value
        for i in range(len(path) - 1, -1, -1):
            cur = path[i]
            cur.visit_count += 1
            cur.value_sum += value_to_backup
            if i > 0:
                reward = path[i].reward_from_parent
                value_to_backup = reward - value_to_backup

        return value_to_backup
