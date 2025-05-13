import torch
import numpy as np


class Evaluator:
    def __init__(self):
        super().__init__()

    def get_sequence_metrics(self, targets, preds, look_back_window):
        metrics = {
            "MSE": mse(targets, preds),
            "MAE": mae(targets, preds),
            # "abs_error": abs_error(targets, preds),
            # "abs_target_sum": abs_target_sum(targets),
            # "abs_target_mean": abs_target_mean(targets),
            # "MAPE": mape(targets, preds),
            # "sMAPE": smape(targets, preds),
            "cross_correlation": correlation(preds, targets),
            "dir_acc": dir_acc(preds, targets, look_back_window),
        }

        # metrics["RMSE"] = np.sqrt(metrics["MSE"])
        # metrics["NRMSE"] = metrics["RMSE"] / metrics["abs_target_mean"]
        # metrics["ND"] = metrics["abs_error"] / metrics["abs_target_sum"]

        return metrics

    def get_metrics(
        self,
        targets: torch.Tensor,
        forecasts: torch.Tensor,
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
            the mean metric values of the batch
        """
        seq_metrics = {}

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
        look_back_window: torch.Tensor = None,
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


def dir_acc(
    preds: np.ndarray, targets: np.ndarray, look_back_window: np.ndarray
) -> float:
    """
    Computes the percentage of correct directional predictions (up/down movement)
    between the last observed value and the predicted/true future value.

    Args:
        preds: Model predictions shape (B, T, C)
        targets: Ground truth values shape (B, T, C)
        look_back_window: Input history window shape (B, L, C)

    Returns:
        Accuracy score between 0 and 1 where 1 = perfect direction prediction
    """
    _, _, C = targets.shape
    last_look_back = look_back_window[
        :, -1, :C
    ]  # filter out the activity info channels
    ground_truth_slope = np.sign(targets[:, 0, :] - last_look_back)  # (B, 1, C)
    prediction_slope = np.sign(preds[:, 0, :] - last_look_back)  # (B, 1, C)

    return np.mean(ground_truth_slope == prediction_slope)


def correlation(preds: np.ndarray, targets: np.ndarray) -> float:
    """
    Computes the mean Pearson correlation between predictions and targets
    across batch and channels.

    Args:
        preds (np.ndarray): Predicted values, shape (B, T, C)
        targets (np.ndarray): Ground truth values, shape (B, T, C)

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


def mae(target: np.ndarray, forecast: np.ndarray) -> float:
    return np.mean(np.abs(target - forecast))


def mse(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mse = mean((Y - \hat{Y})^2)
    """
    return np.mean(np.square(target - forecast))


def abs_error(target: np.ndarray, forecast: np.ndarray) -> float:
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


def mape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        mape = mean(|Y - \hat{Y}| / |Y|))

    See [HA21]_ for more details.
    """
    return np.mean(np.abs(target - forecast) / np.abs(target))


def smape(target: np.ndarray, forecast: np.ndarray) -> float:
    r"""
    .. math::

        smape = 2 * mean(|Y - \hat{Y}| / (|Y| + |\hat{Y}|))

    See [HA21]_ for more details.
    """
    return 2 * np.mean(np.abs(target - forecast) / (np.abs(target) + np.abs(forecast)))


if __name__ == "__main__":
    preds = np.array([1, 2, 3]).reshape(1, -1, 1)
    targets = np.array([-1, 3, 4]).reshape(1, -1, 1)
    look_back_window = np.array([2]).reshape(1, -1, 1)
    import pdb

    preds = np.random.normal(size=(1, 3, 2))
    targets = np.random.normal(size=(1, 3, 2))
    look_back_window = np.random.normal(size=(1, 3, 4))

    pdb.set_trace()

    print(dir_acc(preds, targets, look_back_window))
