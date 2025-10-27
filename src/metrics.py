# ---------------------------------------------------------------
# This file includes code (evaluation functionality) adapted from the Time-Series-Library:
# https://github.com/thuml/Time-Series-Library
#
# Original license: MIT License
# Copyright (c) THUML
#
# If you use this code, please consider citing the original repo:
# https://github.com/thuml/Time-Series-Library
# ---------------------------------------------------------------


import torch
import numpy as np

from numpy.typing import NDArray
from typing import Optional


class Evaluator:
    def __init__(self):
        super().__init__()

    def get_sequence_metrics(
        self,
        targets: NDArray[np.float32],
        preds: NDArray[np.float32],
        look_back_window: NDArray[np.float32],
    ) -> dict[str, float]:
        metrics = {
            "MSE": mse(targets, preds),
            "MAE": mae(targets, preds),
            "DIRACC": dir_acc_improved_single_step(preds, targets, look_back_window),
            "SMAPE": smape(targets, preds),
            "MAPE": mape(targets, preds),  # including MAPE for MIT-BIH study
            "abs_target_mean": abs_target_mean(targets),
            "naive_mae": naive_mae(targets, look_back_window),
            "macro_mase": mase(targets, preds, naive_mae(targets, look_back_window)),
            # "abs_error": abs_error(targets, preds),
            # "abs_target_sum": abs_target_sum(targets),
            # "cross_correlation": correlation(preds, targets),
            # "dir_acc_full": dir_acc_full_horizon(preds, targets, look_back_window),
        }

        # metrics["RMSE"] = np.sqrt(metrics["MSE"])
        # metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        # metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]

        return metrics

    def get_metrics(
        self,
        targets: NDArray[np.float32],
        forecasts: NDArray[np.float32],
        look_back_window: torch.Tensor = None,
    ):
        """

        Parameters
        ----------
        targets
            groundtruth in (B, T, C)
        forecasts
            forecasts in (B, T, C)
        Returns
        -------
        Dict[String, float]
            the metric values of the batch
        """
        seq_metrics: dict[str, float] = {}

        # Calculate metrics for each sequence in the batch
        for i in range(targets.shape[0]):
            single_seq_metrics = self.get_sequence_metrics(
                np.expand_dims(targets[i], axis=0),
                np.expand_dims(forecasts[i], axis=0),
                np.expand_dims(look_back_window[i], axis=0),
            )
            for metric_name, metric_value in single_seq_metrics.items():
                if metric_name not in seq_metrics:
                    seq_metrics[metric_name] = []
                seq_metrics[metric_name].append(metric_value)
        return seq_metrics

    def __call__(
        self,
        targets,
        forecasts,
        look_back_window: Optional[torch.Tensor] = None,
    ):
        """

        Parameters
        ----------
        targets
            groundtruth in (batch_size, prediction_length, target_dim)
        forecasts
            forecasts in (batch_size, num_samples, prediction_length, target_dim)
        Returns
        -------
        Dict[String, float]
            metrics
        """
        targets = targets.cpu().detach().numpy()
        forecasts = forecasts.cpu().detach().numpy()
        look_back_window = look_back_window.cpu().detach().numpy()

        metrics = self.get_metrics(targets, forecasts, look_back_window)
        mean_metrics = metrics.copy()

        for metric_name, metric_value in mean_metrics.items():
            mean_metrics[metric_name] = np.mean(metric_value)

        return mean_metrics, metrics


def dir_acc_improved_single_step(
    preds: NDArray[np.float32],
    targets: NDArray[np.float32],
    look_back_window: NDArray[np.float32],
    epsilon: float = 1e-6,
) -> float:
    """
    Computes the percentage of correct directional predictions (up/down/no movement)
    between the last observed value and the predicted/true future value.

    Args:
        preds: Model predictions shape (B, T, C)
        targets: Ground truth values shape (B, T, C)
        look_back_window: Input history window shape (B, L, C)
        epsilon: A small tolerance to consider a change as 'zero'.

    Returns:
        Accuracy score between 0 and 1 where 1 = perfect direction prediction
    """
    _, _, C = targets.shape
    last_look_back = look_back_window[:, -1, :C]

    # Calculate actual changes
    true_change = targets[:, 0, :] - last_look_back
    pred_change = preds[:, 0, :] - last_look_back

    # Determine direction based on epsilon
    # +1 for increase, -1 for decrease, 0 for no significant change
    ground_truth_direction = np.zeros_like(true_change, dtype=int)
    ground_truth_direction[true_change > epsilon] = 1
    ground_truth_direction[true_change < -epsilon] = -1

    prediction_direction = np.zeros_like(pred_change, dtype=int)
    prediction_direction[pred_change > epsilon] = 1
    prediction_direction[pred_change < -epsilon] = -1

    return np.mean(ground_truth_direction == prediction_direction)


def dir_acc_full_horizon(
    preds: NDArray[np.float32],
    targets: NDArray[np.float32],
    look_back_window: NDArray[np.float32],
    epsilon: float = 1e-6,
) -> float:
    """
    Computes the percentage of correct directional predictions (up/down/no movement)
    for each step within the prediction horizon.

    Args:
        preds: Model predictions shape (B, T, C)
        targets: Ground truth values shape (B, T, C)
        look_back_window: Input history window shape (B, L, C)
        epsilon: A small tolerance to consider a change as 'zero'.

    Returns:
        Accuracy score between 0 and 1 where 1 = perfect direction prediction
    """
    _, _, C = targets.shape

    # Initialize previous values for directional calculation
    # For the first step (t=0) of the prediction, the 'previous' value is the last look-back value.
    prev_targets = np.concatenate(
        (look_back_window[:, -1:, :C], targets[:, :-1, :]), axis=1
    )
    prev_preds = np.concatenate(
        (look_back_window[:, -1:, :C], preds[:, :-1, :]), axis=1
    )

    # Calculate changes for all steps in the prediction horizon
    true_changes = targets - prev_targets
    pred_changes = preds - prev_preds

    # Determine direction based on epsilon for true changes
    ground_truth_direction = np.zeros_like(true_changes, dtype=int)
    ground_truth_direction[true_changes > epsilon] = 1
    ground_truth_direction[true_changes < -epsilon] = -1

    # Determine direction based on epsilon for predicted changes
    prediction_direction = np.zeros_like(pred_changes, dtype=int)
    prediction_direction[pred_changes > epsilon] = 1
    prediction_direction[pred_changes < -epsilon] = -1

    # Compare directions across all elements and calculate mean accuracy
    return np.mean(ground_truth_direction == prediction_direction)


def correlation(preds: NDArray[np.float32], targets: NDArray[np.float32]) -> float:
    """
    Computes the mean Pearson correlation between predictions and targets
    across batch and channels.

    Args:
        preds (NDArray[np.float32]): Predicted values, shape (B, T, C)
        targets (NDArray[np.float32]): Ground truth values, shape (B, T, C)

    Returns:
        float: Mean Pearson correlation over (B, C)
    """
    preds_mean = np.mean(preds, axis=1, keepdims=True)
    targets_mean = np.mean(targets, axis=1, keepdims=True)

    preds_centered = preds - preds_mean
    targets_centered = targets - targets_mean

    numerator = np.sum(preds_centered * targets_centered, axis=1)  # shape (B,C)
    denominator = np.sqrt(
        np.sum(preds_centered**2, axis=1) * np.sum(targets_centered**2, axis=1)
    )  # shape (B, C)

    corr = numerator / (denominator + 1e-8)  # shape: (B, C)

    mean_corr = np.mean(corr)  # scalar

    return mean_corr


def naive_mae(
    target: NDArray[np.float32], look_back_window: NDArray[np.float32]
) -> float:
    last_value = look_back_window[:, -1:, : target.shape[-1]]
    diff = mae(target, last_value)

    return diff


def mae(target: NDArray[np.float32], forecast: NDArray[np.float32]) -> float:
    return np.mean(np.abs(target - forecast))


def mse(target: NDArray[np.float32], forecast: NDArray[np.float32]) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: NDArray[np.float32], forecast: NDArray[np.float32]) -> float:
    r"""
    .. math::

        abs\_error = sum(|Y - \hat{Y}|)
    """
    return np.sum(np.abs(target - forecast))


def abs_target_sum(target) -> float:
    r"""
    .. math::

        abs\_target\_sum = sum(|Y|)
    """
    return np.sum(np.abs(target))


def abs_target_mean(target) -> float:
    r"""
    .. math::

        abs\_target\_mean = mean(|Y|)
    """
    return np.mean(np.abs(target))


def mase(
    target: NDArray[np.float32],
    forecast: NDArray[np.float32],
    naive_error: float,
) -> float:
    r"""
    .. math::

        mase = mean(|Y - \hat{Y}|) / seasonal\_error

    See [HA21]_ for more details.
    """

    EPS = 1e-8
    if naive_error < EPS:
        return 1.0
    diff = np.mean(np.abs(target - forecast), axis=1)
    mase = diff / np.maximum(naive_error, EPS)
    return float(np.mean(mase))


def mape(target: NDArray[np.float32], forecast: NDArray[np.float32]) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / (np.abs(target) + 1e-6))


def smape(target: NDArray[np.float32], forecast: NDArray[np.float32]) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(np.abs(target - forecast) / (np.abs(target) + np.abs(forecast)))


if __name__ == "__main__":
    preds = np.array([1, 2, 3]).reshape(1, -1, 1)
    targets = np.array([-1, 3, 2]).reshape(1, -1, 1)
    look_back_window = np.array([2]).reshape(1, -1, 1)
    import matplotlib.pyplot as plt

    time = list(range(4))
    plt.plot(time, np.concatenate((look_back_window[0, :, 0], preds[0, :, 0])))
    plt.plot(time, np.concatenate((look_back_window[0, :, 0], targets[0, :, 0])))
    plt.show()

    # preds = np.random.normal(size=(1, 3, 2))
    # targets = np.random.normal(size=(1, 3, 2))
    # look_back_window = np.random.normal(size=(1, 3, 4))

    print(dir_acc_full_horizon(preds, targets, look_back_window))
    print(dir_acc_improved_single_step(preds, targets, look_back_window))
