from typing import Protocol

import torch


class Encoder(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, C, 8, 8] -> [B, 64, d_s]"""
        ...


class SmolgenBias(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, num_heads, 64, 64]"""
        ...


class RelativePositionBias(Protocol):
    def forward(self) -> torch.Tensor:
        """-> [num_heads, 64, 64] topology-aware position biases"""
        ...


class PolicyBackbone(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, 64, d_s]"""
        ...


class PolicyHead(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, 64, 64]"""
        ...


class ValueHead(Protocol):
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """[B, 64, d_s] -> (wdl [B, 3], ply [B, 1])
        WDL probabilities sum to 1. Ply is predicted game length.
        """
        ...


class WorldModel(Protocol):
    def forward(
        self,
        states: torch.Tensor,
        action_from: torch.Tensor,
        action_to: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        states: [B, T, 64, d_s], action_from: [B, T], action_to: [B, T]
        -> next_states [B, T, 64, d_s], rewards [B, T]
        """
        ...


class DiffusionModule(Protocol):
    def forward(
        self, x: torch.Tensor, t: torch.Tensor, cond: torch.Tensor
    ) -> torch.Tensor:
        """
        x: [B, 64, d_s] noisy latent
        t: [B] timestep indices
        cond: [B, 64, d_s] condition
        -> [B, 64, d_s] predicted noise
        """
        ...


class ConsistencyProjector(Protocol):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """[B, 64, d_s] -> [B, proj_dim]"""
        ...
