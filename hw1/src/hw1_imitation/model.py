"""Model definitions for Push-T imitation policies."""

from __future__ import annotations

import abc
from typing import Literal, TypeAlias

import torch
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        dims = (state_dim, *hidden_dims, chunk_size * action_dim)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        return nn.functional.mse_loss(self.sample_actions(state), action_chunk)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        return self.net(state).reshape(
            *state.shape[:-1], self.chunk_size, self.action_dim
        )


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        action_chunk_dim = chunk_size * action_dim
        dims = (state_dim + action_chunk_dim + 1, *hidden_dims, action_chunk_dim)
        layers: list[nn.Module] = []
        for index, (input_dim, output_dim) in enumerate(
            zip(dims[:-1], dims[1:], strict=True)
        ):
            layers.append(nn.Linear(input_dim, output_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        noise = torch.randn_like(action_chunk)
        flow_time = torch.rand(
            batch_size,
            1,
            1,
            device=action_chunk.device,
            dtype=action_chunk.dtype,
        )
        noisy_chunk = (1.0 - flow_time) * noise + flow_time * action_chunk
        target_velocity = action_chunk - noise
        model_input = torch.cat(
            (state, noisy_chunk.flatten(start_dim=1), flow_time.flatten(start_dim=1)),
            dim=1,
        )
        predicted_velocity = self.net(model_input).reshape_as(action_chunk)
        return nn.functional.mse_loss(predicted_velocity, target_velocity)

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        if num_steps <= 0:
            raise ValueError("num_steps must be positive")

        batch_size = state.shape[0]
        action_chunk = torch.randn(
            batch_size,
            self.chunk_size,
            self.action_dim,
            device=state.device,
            dtype=state.dtype,
        )
        step_size = 1.0 / num_steps

        for step in range(num_steps):
            flow_time = torch.full(
                (batch_size, 1),
                step * step_size,
                device=state.device,
                dtype=state.dtype,
            )
            model_input = torch.cat(
                (state, action_chunk.flatten(start_dim=1), flow_time), dim=1
            )
            velocity = self.net(model_input).reshape_as(action_chunk)
            action_chunk = action_chunk + step_size * velocity

        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
