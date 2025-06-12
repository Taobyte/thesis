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
import lightning as L
import numpy as np

from omegaconf import DictConfig
from typing import Dict, Tuple, Union
from collections import defaultdict

from src.metrics import Evaluator
from src.plotting import (
    plot_max_min_median_predictions,
    plot_metric_histograms,
    plot_entire_series,
)


def get_model_kwargs(config: DictConfig, datamodule: L.LightningDataModule) -> dict:
    model_kwargs = {}
    if config.model.name in ["gp", "dklgp"]:
        model_kwargs["inducing_points"] = datamodule.get_inducing_points(
            config.model.n_points, config.model.strategy
        )
        model_kwargs["train_dataset_length"] = datamodule.get_train_dataset_length()
    elif config.model.name in ["exactgp", "xgboost"]:
        lbw_train_dataset, pw_train_dataset = datamodule.get_train_dataset()
        model_kwargs["lbw_train_dataset"] = lbw_train_dataset
        model_kwargs["pw_train_dataset"] = pw_train_dataset
        lbw_val_dataset, pw_val_dataset = datamodule.get_val_dataset()
        model_kwargs["lbw_val_dataset"] = lbw_val_dataset
        model_kwargs["pw_val_dataset"] = pw_val_dataset

    return model_kwargs


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
        x_norm = (x - mean[:, :, :C]) / (std[:, :, :C] + 1e-8)

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
    x_denorm = x_norm.clone()
    x_denorm = x_denorm[:, :, :C] * std[:, :, :C] + mean[:, :, :C]
    return x_denorm.float()


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        wandb_project: str = "c_keusch/thesis",
        n_trials: int = 10,
        tune: bool = False,
        name: str = None,
        use_plots: bool = False,
    ):
        super().__init__()

        self.n_trials = n_trials
        self.wandb_project = wandb_project
        self.tune = tune
        self.name = name
        self.use_plots = use_plots

        self.evaluator = Evaluator()

        self.look_back_channel_dim = None
        self.target_channel_dim = None
        self.use_static_features = None
        self.use_dynamic_features = None
        self.static_exogenous_variables = None
        self.dynamic_exogenous_variables = None

    @property
    def has_probabilistic_forecast(self) -> bool:
        """
        Indicates whether the model provides probabilistic forecasts (mean and std).
        This property should be overridden in subclasses for probabilistic models.
        """
        return self.name in [
            "gp",
            "dklgp",
            "exactgp",
            "bnn",
        ]

    def model_forward(
        self, look_back_window: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Performs a forward pass through the model to generate predictions for the
        prediction window.

        The behavior and return type of this method depend on whether the model
        is a point forecaster or a probabilistic forecaster.

        Args:
            look_back_window (torch.Tensor): A tensor representing the
                lookback window. Its shape is expected to be (B, T_lb, C_lb),
                where B is the batch size, T_lb is the length of the lookback
                window, and C_lb is the number of input channels.

        Returns:
            Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
                - If the model is a **point forecaster** (e.g., `self.has_probabilistic_forecast` is False):
                  Returns a single `torch.Tensor` of shape (B, L, C),
                  representing the point forecast (mean prediction) for the
                  prediction window.
                - If the model is a **probabilistic forecaster** (e.g., `self.has_probabilistic_forecast` is True):
                  Returns a `tuple` containing two `torch.Tensor`s:
                    - The first tensor is the **mean prediction** for the
                      prediction window, of shape (B, L, C).
                    - The second tensor is the **standard deviation** (or
                      some other measure of uncertainty, e.g., variance or
                      log-variance) of the predictions for the prediction window,
                      also of shape (B, L, C).
        """
        raise NotImplementedError

    def model_specific_train_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> float:
        raise NotImplementedError

    def model_specific_val_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> float:
        raise NotImplementedError

    def on_fit_start(self):
        datamodule = self.trainer.datamodule
        self.look_back_channel_dim = datamodule.look_back_channel_dim
        self.target_channel_dim = datamodule.target_channel_dim
        self.use_static_features = datamodule.use_static_features
        self.use_dynamic_features = datamodule.use_dynamic_features
        self.static_exogenous_variables = datamodule.static_exogenous_variables
        self.dynamic_exogenous_variables = datamodule.dynamic_exogenous_variables

        self.local_norm_channels = datamodule.local_norm_channels

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch

        look_back_window_norm, mean, std = local_z_norm(
            look_back_window, self.local_norm_channels
        )
        prediction_window_norm, _, _ = local_z_norm(
            prediction_window, self.local_norm_channels, mean, std
        )

        loss = self.model_specific_train_step(
            look_back_window_norm, prediction_window_norm
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch

        look_back_window_norm, mean, std = local_z_norm(
            look_back_window, self.local_norm_channels
        )
        prediction_window_norm, _, _ = local_z_norm(
            prediction_window, self.local_norm_channels, mean, std
        )

        loss = self.model_specific_val_step(
            look_back_window_norm, prediction_window_norm
        )

        return loss

    def test_step(self, batch, batch_idx):
        metrics, current_metrics = self.evaluate(batch, batch_idx)

        for k, v in current_metrics.items():
            self.metric_full[k] += v
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.batch_size = []

        self.metric_full = defaultdict(list)

    def on_test_epoch_end(self):
        # log the average metrics to wandb
        avg_metrics = self.calculate_weighted_average(
            self.metrics_dict, self.batch_size
        )
        self.log_dict(avg_metrics, logger=True, sync_dist=True)

        # plot if use_plots is true and process rank equals to 0 (multi gpu training)
        if self.use_plots and self.trainer.is_global_zero:
            # plot metric histograms
            plot_metric_histograms(self.logger, self.metric_full)

            # plot the whole timeseries with the metrics
            datamodule = self.trainer.datamodule
            plot_entire_series(
                self.logger,
                datamodule,
                self.metric_full,
            )

            # plot best, worst and median prediction for each metric
            plot_max_min_median_predictions(self)

    def evaluate(self, batch, batch_idx):
        look_back_window, prediction_window = batch
        self.batch_size.append(look_back_window.shape[0])

        look_back_window_norm, mean, std = local_z_norm(
            look_back_window, self.local_norm_channels
        )

        # Prediction
        if self.has_probabilistic_forecast:
            preds, _ = self.model_forward(look_back_window_norm)
        else:
            preds = self.model_forward(look_back_window_norm)

        preds = preds[:, :, : prediction_window.shape[-1]]

        assert preds.shape == prediction_window.shape

        # Metric Calculation
        denormalized_preds = local_z_denorm(preds, self.local_norm_channels, mean, std)
        metrics, current_metrics = self.evaluator(
            prediction_window, denormalized_preds, look_back_window
        )
        self.update_metrics(metrics)

        return metrics, current_metrics

    def update_metrics(self, new_metrics: Dict):
        prefix = "test"
        for metric_name, metric_value in new_metrics.items():
            metric_key = f"{prefix}_{metric_name}"
            if metric_key not in self.metrics_dict:
                self.metrics_dict[metric_key] = []
            self.metrics_dict[metric_key].append(metric_value)

    def calculate_weighted_average(self, metrics_dict: Dict, batch_size: list):
        metrics = {}
        for key, value in metrics_dict.items():
            metrics[key] = np.sum(value * np.array(batch_size)) / np.sum(batch_size)
        return metrics
