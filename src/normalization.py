import torch
import numpy as np

from typing import Tuple


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


def local_z_norm(
    x: torch.Tensor,
    local_norm_channels: int,
    mean: torch.Tensor = None,
    std: torch.Tensor = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Applies local Z-normalization to the first `local_norm_channels` of a
    time series tensor.

    This function can operate in two modes:
    1. If `mean` and `std` are not provided, it calculates them from the
       input tensor `x` (assumed to be a 'lookback' window) and applies
       the normalization.
    2. If `mean` and `std` are provided, it uses them to normalize `x`
       (assumed to be a 'prediction' window).

    The normalization is only applied to the specified number of dynamic
    time series channels, leaving static features (e.g., one-hot encoded
    activity info, age, weight, height) untouched.

    Args:
        x (torch.Tensor): The input tensor with shape (batch_size, sequence_length, total_channels).
                          It contains both dynamic time series channels and static features.
        local_norm_channels (int): The number of initial channels in `x` that
                                   represent dynamic time series data and should be normalized.
                                   The remaining channels are assumed to be static and pre-normalized.
        mean (Optional[torch.Tensor]): Optional. The mean tensor, typically calculated from a
                                       lookback window, to use for normalization. If None,
                                       the mean is calculated from `x`.
                                       Shape should be (batch_size, 1, local_norm_channels).
        std (Optional[torch.Tensor]): Optional. The standard deviation tensor, typically
                                      calculated from a lookback window, to use for normalization.
                                      If None, the standard deviation is calculated from `x`.
                                      Shape should be (batch_size, 1, local_norm_channels).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            - x_norm (torch.Tensor): The normalized tensor (float, detached from graph).
            - mean (torch.Tensor): The mean tensor used for normalization (float, detached from graph).
            - std (torch.Tensor): The standard deviation tensor used for normalization (float, detached from graph).
    """
    if mean is None or std is None:
        # here we normalize the look back window
        _, _, C = x.shape

        assert local_norm_channels <= C
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
        x_norm = x.clone()
        x_norm[:, :, :local_norm_channels] = (
            x_norm[:, :, :local_norm_channels] - mean[:, :, :local_norm_channels]
        ) / (std[:, :, :local_norm_channels] + 1e-8)
    else:
        # here we normalize the prediction window

        _, _, C = x.shape
        zero_std_mask = std[:, :, :C] < 1e-7
        diff = x - mean[:, :, :C]
        x_norm = (x - mean[:, :, :C]) / (std[:, :, :C] + 1e-8)
        x_norm = torch.where(zero_std_mask, diff, x_norm)

    return x_norm.float().detach(), mean.float().detach(), std.float().detach()


def local_z_denorm(
    x_norm: torch.Tensor,
    local_norm_channels: int,
    mean: torch.Tensor,
    std: torch.Tensor,
) -> torch.Tensor:
    """
    Applies local Z-denormalization (reverses Z-normalization) to a tensor,
    using provided mean and standard deviation.

    This function denormalizes the channels that were previously normalized.
    It's designed to work for both lookback and prediction windows, by ensuring
    that it only processes up to `local_norm_channels` or the actual number
    of channels in `x_norm`, whichever is smaller. The static features, which
    were not normalized, remain untouched as they are outside the `local_norm_channels` range.

    Args:
        x_norm (torch.Tensor): The input tensor that was previously Z-normalized,
                            with shape (batch_size, sequence_length, total_channels).
        local_norm_channels (int): The number of initial channels in `x_norm` that
                                represent dynamic time series data and were normalized.
                                The remaining channels are assumed to be static.
        mean (torch.Tensor): The mean tensor used during the original normalization.
                            Shape should be compatible, e.g., (batch_size, 1, num_normalized_channels).
        std (torch.Tensor): The standard deviation tensor used during the original normalization.
                            Shape should be compatible, e.g., (batch_size, 1, num_normalized_channels).

    Returns:
        torch.Tensor: The denormalized tensor (float type).
    """
    _, _, C = x_norm.shape
    C = min(
        C, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be C
    zero_std_mask = std[:, :, :C] < 1e-7
    x_denorm = x_norm.clone()
    sum = x_denorm[:, :, :C] + mean[:, :, :C]
    x_denorm = x_denorm[:, :, :C] * std[:, :, :C] + mean[:, :, :C]
    x_denorm = torch.where(zero_std_mask, sum, x_denorm)
    return x_denorm.float()
