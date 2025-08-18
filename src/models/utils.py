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


import lightning as L
import numpy as np

from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Any
from collections import defaultdict

from src.metrics import Evaluator
from src.normalization import global_z_denorm, undo_differencing, min_max_denorm
from src.datasets.utils import BaseDataModule
from src.plotting import plot_max_min_median_predictions


def get_model_kwargs(config: DictConfig, datamodule: BaseDataModule) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    if config.model.name in ["gp", "dklgp"]:
        model_kwargs["inducing_points"] = datamodule.get_inducing_points(
            config.model.n_points, config.model.strategy
        )
        model_kwargs["train_dataset_length"] = datamodule.get_train_dataset_length()
    elif config.model.name in config.numpy_models:
        lbw_train_dataset, pw_train_dataset = datamodule.get_train_dataset()
        lbw_val_dataset, pw_val_dataset = datamodule.get_val_dataset()
        model_kwargs["lbw_train_dataset"] = lbw_train_dataset
        model_kwargs["pw_train_dataset"] = pw_train_dataset
        model_kwargs["lbw_val_dataset"] = lbw_val_dataset
        model_kwargs["pw_val_dataset"] = pw_val_dataset

    return model_kwargs


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        wandb_project: str = "c_keusch/thesis",
        n_trials: int = 10,
        name: str = "timesnet",
        use_plots: bool = False,
        normalization: str = "local",
        tune: bool = False,
        probabilistic_models: list[str] = [],
        experiment_name: str = "endo_only",
        seed: int = 0,
        return_whole_series: bool = False,
    ):
        super().__init__()

        self.n_trials = n_trials
        self.wandb_project = wandb_project
        self.name = name
        self.tune = tune
        self.use_plots = use_plots
        self.normalization = normalization
        self.probabilistic_forecast_models = probabilistic_models
        self.experiment_name = experiment_name
        self.seed = seed
        self.return_whole_series = return_whole_series

        self.evaluator = Evaluator()

    @property
    def has_probabilistic_forecast(self) -> bool:
        """
        Indicates whether the model provides probabilistic forecasts (mean and std).
        This property should be overridden in subclasses for probabilistic models.
        """
        return self.name in self.probabilistic_forecast_models

    def model_forward(self, look_back_window: Tensor) -> Union[Tensor, Any]:
        """
        Performs a forward pass through the model to generate predictions for the
        prediction window.

        The behavior and return type of this method depend on whether the model
        is a point forecaster or a probabilistic forecaster.

        Args:
            look_back_window (Tensor): A tensor representing the
                lookback window. Its shape is expected to be (B, T_lb, C_lb),
                where B is the batch size, T_lb is the length of the lookback
                window, and C_lb is the number of input channels.

        Returns:
            Union[Tensor, Tuple[Tensor, Tensor]]:
                - If the model is a **point forecaster** (e.g., `self.has_probabilistic_forecast` is False):
                  Returns a single `Tensor` of shape (B, L, C),
                  representing the point forecast (mean prediction) for the
                  prediction window.
                - If the model is a **probabilistic forecaster** (e.g., `self.has_probabilistic_forecast` is True):
                  Returns a `tuple` containing two `Tensor`s:
                    - The first tensor is the **mean prediction** for the
                      prediction window, of shape (B, L, C).
                    - The second tensor is the **standard deviation** (or
                      some other measure of uncertainty, e.g., variance or
                      log-variance) of the predictions for the prediction window,
                      also of shape (B, L, C).
        """
        raise NotImplementedError

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        raise NotImplementedError

    def on_fit_start(self):
        datamodule: BaseDataModule = self.trainer.datamodule  # type: ignore
        self.look_back_window = datamodule.look_back_window
        self.prediction_window = datamodule.prediction_window
        self.look_back_channel_dim: int = datamodule.look_back_channel_dim
        self.target_channel_dim: int = datamodule.target_channel_dim
        self.use_static_features: bool = datamodule.use_static_features
        self.use_dynamic_features: bool = datamodule.use_dynamic_features
        self.static_exogenous_variables: int = datamodule.static_exogenous_variables
        self.dynamic_exogenous_variables: int = datamodule.dynamic_exogenous_variables

        self.local_norm_channels: int = datamodule.local_norm_channels

    def training_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tensor:
        if self.return_whole_series:
            series, lengths = batch
            loss = self.model_specific_train_step(series, lengths)
        else:
            _, look_back_window_norm, prediction_window_norm = batch
            loss = self.model_specific_train_step(
                look_back_window_norm, prediction_window_norm
            )
        return loss

    def validation_step(
        self, batch: Union[Tensor, Tuple[Tensor, Tensor, Tensor]], batch_idx: int
    ) -> Tensor:
        if self.return_whole_series:
            series, lengths = batch
            loss = self.model_specific_val_step(series, lengths)
        else:
            _, look_back_window_norm, prediction_window_norm = batch
            loss = self.model_specific_val_step(
                look_back_window_norm, prediction_window_norm
            )
        return loss

    def test_step(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> dict[str, float]:
        metrics, current_metrics = self.evaluate(batch, batch_idx)

        for k, v in current_metrics.items():
            self.metric_full[k] += v
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict: dict[str, list[float]] = {}
        self.batch_size: list[int] = []

        self.metric_full: defaultdict[str, list[float]] = defaultdict(list[float])

    def on_test_epoch_end(self):
        # log the average metrics to wandb
        avg_metrics = self.calculate_weighted_average(
            self.metrics_dict, self.batch_size
        )

        avg_metrics["RMSE"] = avg_metrics["MSE"] ** 0.5
        avg_metrics["NRMSE"] = avg_metrics["RMSE"] / avg_metrics["abs_target_mean"]
        avg_metrics["ND"] = avg_metrics["MAE"] / avg_metrics["abs_target_mean"]
        avg_metrics["MASE"] = avg_metrics["MAE"] / avg_metrics["naive_mae"]
        self.log_dict(avg_metrics, logger=True, sync_dist=True)

        if self.use_plots:
            plot_max_min_median_predictions(self)

    def evaluate(
        self, batch: Tuple[Tensor, Tensor, Tensor], batch_idx: int
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        (
            look_back_window,
            look_back_window_norm,
            prediction_window_norm,
        ) = batch
        batch_size: int = look_back_window.shape[0]
        self.batch_size.append(batch_size)
        # Prediction
        if self.has_probabilistic_forecast:
            preds, _ = self.model_forward(look_back_window_norm)
        else:
            preds = self.model_forward(look_back_window_norm)

        preds = preds[:, :, : self.target_channel_dim]
        assert preds.shape == prediction_window_norm.shape

        if self.normalization == "global":
            device = look_back_window_norm.device
            train_dataset: Dataset = self.trainer.datamodule.train_dataset  # type:ignore
            mean = train_dataset.mean
            std = train_dataset.std
            mean = Tensor(mean).reshape(1, 1, -1).to(device).float()
            std = Tensor(std).reshape(1, 1, -1).to(device).float()
            preds = global_z_denorm(preds, self.local_norm_channels, mean, std)
            prediction_window = global_z_denorm(
                prediction_window_norm, self.local_norm_channels, mean, std
            )

        elif self.normalization == "minmax":
            device = look_back_window_norm.device
            train_dataset: Dataset = self.trainer.datamodule.train_dataset  # type:ignore
            min = train_dataset.min
            max = train_dataset.max
            min = Tensor(min).reshape(1, 1, -1).to(device).float()
            max = Tensor(max).reshape(1, 1, -1).to(device).float()
            preds = min_max_denorm(preds, self.local_norm_channels, min, max)
            prediction_window = min_max_denorm(
                prediction_window_norm, self.local_norm_channels, min, max
            )

        elif self.normalization == "difference":
            preds = undo_differencing(look_back_window, preds)
            prediction_window = undo_differencing(
                look_back_window, prediction_window_norm
            )
        else:
            prediction_window = prediction_window_norm

        # Metric Calculation
        metrics, current_metrics = self.evaluator(
            prediction_window, preds, look_back_window
        )
        self.update_metrics(metrics)

        return metrics, current_metrics

    def update_metrics(self, new_metrics: Dict):
        for metric_name, metric_value in new_metrics.items():
            if metric_name not in self.metrics_dict:
                self.metrics_dict[metric_name] = []
            self.metrics_dict[metric_name].append(metric_value)

    def calculate_weighted_average(
        self, metrics_dict: dict[str, list[float]], batch_size: list[int]
    ) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, value in metrics_dict.items():
            metrics[key] = np.sum(value * np.array(batch_size)) / np.sum(batch_size)

        return metrics
