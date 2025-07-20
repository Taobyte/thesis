import torch
import numpy as np


def global_z_norm(
    x: torch.Tensor, local_norm_channels: int, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    device = x.device
    mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
    std = torch.tensor(std).reshape(1, 1, -1).to(device).float()

    _, _, C = x.shape
    C = min(
        C, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be C
    x_static = x[:, :, C:]
    x_norm = (x[:, :, :C] - mean[:, :, :C]) / (
        std[:, :, :C] + 1e-8
    )  # TODO: THIS IS NOT CORRECT

    combined = torch.concat((x_norm, x_static), dim=2)

    return combined


def global_z_denorm(
    x: torch.Tensor, local_norm_channels: int, mean: np.ndarray, std: np.ndarray
) -> torch.Tensor:
    device = x.device
    mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
    std = torch.tensor(std).reshape(1, 1, -1).to(device).float()

    _, _, C = x.shape
    C = min(
        C, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be C
    x_static = x[:, :, C:]
    x_denorm = x[:, :, :C] * std[:, :, :C] + mean[:, :, :C]
    combined = torch.concat((x_denorm, x_static), dim=2)
    return combined


def local_z_norm_numpy(batch: np.ndarray, mean=None, std=None):
    if mean is None and std is None:
        mean = np.mean(batch, axis=1, keepdims=True)
        std = np.std(batch, axis=1, keepdims=True) + 1e-5
    normed = (batch - mean) / std
    return normed, mean, std
