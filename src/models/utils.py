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


import math
import torch
import lightning as L
import numpy as np


from typing import Dict, Tuple
from collections import defaultdict

from src.metrics import Evaluator
from src.plotting import plot_prediction_wandb


def z_normalization(x, global_mean: torch.Tensor, global_std: torch.Tensor):
    B, T, C = x.shape
    eps = 1e-8
    x_norm = (x - global_mean[:, :, :C]) / (global_std[:, :, :C] + eps)

    return x_norm.float()


def z_denormalization(x_norm, global_mean: torch.Tensor, global_std: torch.Tensor):
    B, T, C = x_norm.shape
    x_denorm = x_norm * global_std[:, :, :C] + global_mean[:, :, :C]
    return x_denorm.float()


def local_z_norm(
    x: torch.Tensor, mean: torch.Tensor = None, std: torch.Tensor = None
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    _, _, C = (
        x.shape
    )  # we need channel dim, because channel dims vary for look_back_window and prediction_window
    if mean is None or std is None:
        mean = x.mean(dim=1, keepdim=True)
        std = x.std(dim=1, keepdim=True)
    x_norm = (x - mean[:, :, :C]) / (std[:, :, :C] + 1e-8)
    return x_norm.float(), mean.float(), std.float()


def local_z_denorm(
    x_norm: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    _, _, C = x_norm.shape
    x_denorm = x_norm * std[:, :, :C] + mean[:, :, :C]
    return x_denorm.float()


class BaseLightningModule(L.LightningModule):
    def __init__(self):
        super().__init__()

        self.evaluator = Evaluator()

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

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch

        look_back_window_norm, mean, std = local_z_norm(look_back_window)
        prediction_window_norm, _, _ = local_z_norm(prediction_window, mean, std)

        loss = self.model_specific_train_step(
            look_back_window_norm, prediction_window_norm
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch

        look_back_window_norm, mean, std = local_z_norm(look_back_window)
        prediction_window_norm, _, _ = local_z_norm(prediction_window, mean, std)

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

        # use_heart_rate = self.trainer.datamodule.use_heart_rate
        # use_activity_info = self.trainer.datamodule.use_activity_info

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
                target = target.unsqueeze(0)
                pred = self.model_forward(look_back_window)[:, :, : target.shape[-1]]

                assert pred.shape == target.shape

                plot_prediction_wandb(
                    look_back_window,
                    target,
                    pred,
                    self.logger,
                    metric_name,
                    v[idx],
                    type,
                    True,
                    25,
                    "Heartrate",
                )

    def evaluate(self, batch, batch_idx):
        look_back_window, prediction_window = batch
        self.batch_size.append(look_back_window.shape[0])

        look_back_window_norm, mean, std = local_z_norm(look_back_window)

        # Prediction
        preds = self.model_forward(look_back_window_norm)

        preds = preds[:, :, : prediction_window.shape[-1]]

        assert preds.shape == prediction_window.shape

        # Metric Calculation
        denormalized_preds = local_z_denorm(preds, mean, std)
        metrics, current_metrics = self.evaluator(prediction_window, denormalized_preds)
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


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lradj: str,
    learning_rate: float,
    train_epochs: int = 10,
):
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "type3":
        lr_adjust = {
            epoch: learning_rate
            if epoch < 3
            else learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif lradj == "cosine":
        lr_adjust = {
            epoch: learning_rate / 2 * (1 + math.cos(epoch / train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # print("Updating learning rate to {}".format(lr))
        return lr
    return optimizer.param_groups[0]["lr"]
