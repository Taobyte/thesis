import torch
import numpy as np

from numpy.typing import NDArray
from typing import Optional, Tuple


def global_z_norm(
    x: torch.Tensor, local_norm_channels: int, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    _, _, c = x.shape
    c = min(
        c, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be c
    x_static = x[:, :, c:]
    x_norm = (x[:, :, :c] - mean[:, :, :c]) / (std[:, :, :c] + 1e-8)

    combined = torch.concat((x_norm, x_static), dim=2)

    return combined


def global_z_denorm(
    x: torch.Tensor, local_norm_channels: int, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    _, _, c = x.shape
    c = min(
        c, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be c
    x_static = x[:, :, c:]
    x_denorm = x[:, :, :c] * std[:, :, :c] + mean[:, :, :c]
    combined = torch.concat((x_denorm, x_static), dim=2)
    return combined


def local_z_norm_numpy(
    batch: NDArray[np.float32],
    mean: Optional[NDArray[np.float32]] = None,
    std: Optional[NDArray[np.float32]] = None,
) -> Tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    if mean is None and std is None:
        mean = np.mean(batch, axis=1, keepdims=True)
        std = np.std(batch, axis=1, keepdims=True) + 1e-5
    assert mean is not None and std is not None
    normed = (batch - mean) / std
    return normed, mean, std


def undo_differencing(look_back_window: torch.Tensor, predicted_deltas: torch.Tensor):
    _, _, c = predicted_deltas.shape
    last_value = look_back_window[:, -1:, :c]
    cumsum_deltas = torch.cumsum(predicted_deltas, dim=1)
    reconstructed = last_value + cumsum_deltas
    return reconstructed
