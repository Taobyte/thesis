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

from src.metrics import Evaluator
from src.plotting import plot_prediction_wandb


def z_normalization(x, global_mean: np.ndarray, global_std: np.ndarray):
    eps = 1e-8
    x_norm = (x - global_mean[None, None, :]) / (global_std[None, None, :] + eps)

    return x_norm.float()


def z_denormalization(x_norm, global_mean: np.ndarray, global_std: np.ndarray):
    x_denorm = x_norm * global_std[None, None, :] + global_mean
    return x_denorm.float()


class BaseLightningModule(L.LightningModule):
    def __init__(self, global_mean: np.ndarray, global_std: np.ndarray):
        super().__init__()

        self.global_mean = global_mean
        self.global_std = global_std
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

    def on_fit_start(self):
        self.global_mean = torch.Tensor(self.global_mean).to(self.device)
        self.global_std = torch.Tensor(self.global_std).to(self.device)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch
        look_back_window_norm = z_normalization(
            look_back_window, self.global_mean, self.global_std
        )
        prediction_window_norm = z_normalization(
            prediction_window, self.global_mean, self.global_std
        )

        loss = self.model_specific_train_step(
            look_back_window_norm, prediction_window_norm
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        # normalize data
        look_back_window, prediction_window = batch
        look_back_window_norm = z_normalization(
            look_back_window, self.global_mean, self.global_std
        )
        prediction_window_norm = z_normalization(
            prediction_window, self.global_mean, self.global_std
        )

        loss = self.model_specific_val_step(
            look_back_window_norm, prediction_window_norm
        )
        return loss

    def test_step(self, batch, batch_idx):
        metrics = self.evaluate(batch)
        return metrics

    def on_test_epoch_start(self):
        self.metrics_dict = {}
        self.batch_size = []

    def on_test_epoch_end(self):
        avg_metrics = self.calculate_weighted_average(
            self.metrics_dict, self.batch_size
        )

        self.log_dict(avg_metrics, logger=True)

    def evaluate(self, batch):
        look_back_window, prediction_window = batch
        self.batch_size.append(look_back_window.shape[0])

        # Normalization
        look_back_window_norm = z_normalization(
            look_back_window, self.global_mean, self.global_std
        )

        # Prediction
        preds = self.model_forward(look_back_window_norm)

        # Metric Calculation
        denormalized_preds = z_denormalization(preds, self.global_mean, self.global_std)
        metrics = self.evaluator(prediction_window, denormalized_preds)
        self.update_metrics(metrics)

        plot_prediction_wandb(
            look_back_window, prediction_window, denormalized_preds, self.logger
        )

        return metrics

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
        print("Updating learning rate to {}".format(lr))
        return lr
    return optimizer.param_groups[0]["lr"]
