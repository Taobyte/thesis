# ---------------------------------------------------------------
# This file includes code adapted from the Time-Series-Library:
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
from typing import Dict, Tuple
from collections import defaultdict

from src.metrics import Evaluator
from src.plotting import plot_prediction_wandb


def get_model_kwargs(config: DictConfig, datamodule: L.LightningDataModule) -> dict:
    model_kwargs = {}
    if config.model.name == "gp":
        model_kwargs["inducing_points"] = datamodule.get_inducing_points(
            config.model.n_points, config.model.strategy
        )
        model_kwargs["train_dataset_length"] = datamodule.get_train_dataset_length()
    elif config.model.name == "xgboost":
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
    _, _, C = x_norm.shape
    C = min(
        C, local_norm_channels
    )  # for look back window, this will be local_norm_channels and for prediction window it will be C
    x_denorm = x_norm.clone()
    x_denorm = x_denorm[:, :, :C] * std[:, :, :C] + mean[:, :, :C]
    return x_denorm.float()


class BaseLightningModule(L.LightningModule):
    def __init__(self, wandb_project: str = "c_keusch/thesis", n_trials: int = 10):
        super().__init__()

        self.n_trials = n_trials
        self.wandb_project = wandb_project

        self.evaluator = Evaluator()

        self.look_back_channel_dim = None
        self.target_channel_dim = None
        self.use_static_features = None
        self.use_dynamic_features = None
        self.static_exogenous_variables = None
        self.dynamic_exogenous_variables = None

    def model_forward(self, look_back_window: torch.Tensor):
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

        self.local_norm_channels = (
            datamodule.target_channel_dim + datamodule.dynamic_exogenous_variables
        )

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
        avg_metrics = self.calculate_weighted_average(
            self.metrics_dict, self.batch_size
        )

        self.log_dict(avg_metrics, logger=True)

        # plot best, worst and median
        for metric_name, v in self.metric_full.items():
            sorted_indices = np.argsort(v)
            min_idx = sorted_indices[0]
            max_idx = sorted_indices[-1]
            median_idx = sorted_indices[len(v) // 2]
            if metric_name == "cross_correlation":
                zipped = zip(
                    ["worst", "best", "median"], [min_idx, max_idx, median_idx]
                )
            else:
                zipped = zip(
                    ["worst", "best", "median"], [max_idx, min_idx, median_idx]
                )

            for type, idx in zipped:
                look_back_window, target = self.trainer.test_dataloaders.dataset[idx]
                look_back_window = look_back_window.unsqueeze(0).to(self.device)
                look_back_window_norm, mean, std = local_z_norm(
                    look_back_window, self.local_norm_channels
                )
                target = target.unsqueeze(0)
                pred = self.model_forward(look_back_window_norm)[
                    :, :, : target.shape[-1]
                ]
                pred_denorm = local_z_denorm(pred, self.local_norm_channels, mean, std)
                assert pred_denorm.shape == target.shape

                if hasattr(self.trainer.datamodule, "use_heart_rate"):
                    use_heart_rate = self.trainer.datamodule.use_heart_rate
                else:
                    use_heart_rate = False

                plot_prediction_wandb(
                    look_back_window,
                    target,
                    pred_denorm,
                    wandb_logger=self.logger,
                    metric_name=metric_name,
                    metric_value=v[idx],
                    type=type,
                    use_heart_rate=use_heart_rate,
                    freq=self.trainer.datamodule.freq,
                    dataset=self.trainer.datamodule.name,
                )

    def evaluate(self, batch, batch_idx):
        look_back_window, prediction_window = batch
        self.batch_size.append(look_back_window.shape[0])

        look_back_window_norm, mean, std = local_z_norm(
            look_back_window, self.local_norm_channels
        )

        # Prediction
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
