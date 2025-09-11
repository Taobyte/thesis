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

from torch import Tensor
from torch.utils.data import Dataset
from omegaconf import DictConfig
from typing import Dict, Tuple, Union, Any
from collections import defaultdict
from dataclasses import dataclass, field

from src.metrics import Evaluator
from src.normalization import global_z_denorm, min_max_denorm
from src.datasets.utils import BaseDataModule


@dataclass
class ModelOutput:
    preds: Tensor
    aux: Dict[str, Any] = field(default_factory=dict)


def get_model_kwargs(config: DictConfig, datamodule: BaseDataModule) -> dict[str, Any]:
    model_kwargs: dict[str, Any] = {}
    if config.model.name in ["gp", "dklgp"]:
        model_kwargs["inducing_points"] = datamodule.get_inducing_points(
            config.model.n_points, config.model.strategy
        )
        model_kwargs["train_dataset_length"] = datamodule.get_train_dataset_length()

    return model_kwargs


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        name: str,
        local_norm: str,
        n_trials: int,
        use_plots: bool,
        normalization: str,
        local_norm_endo_only: bool,
        tune: bool,
        probabilistic_models: list[str],
        experiment_name: str,
        seed: int,
        return_whole_series: bool,
    ):
        super().__init__()

        self.n_trials = n_trials
        self.name = name
        self.tune = tune
        self.use_plots = use_plots
        self.normalization = normalization
        self.local_norm = local_norm
        self.local_norm_endo_only = local_norm_endo_only
        self.probabilistic_forecast_models = probabilistic_models
        self.experiment_name = experiment_name
        self.seed = seed
        self.return_whole_series = return_whole_series

        self.evaluator = Evaluator()

    @property
    def has_probabilistic_forecast(self) -> bool:
        return self.name in self.probabilistic_forecast_models

    def _call_model(self, x: Tensor) -> ModelOutput:
        if self.has_probabilistic_forecast:
            out, _ = self.model_specific_forward(x)
        else:
            out = self.model_specific_forward(x)

        if isinstance(out, ModelOutput):
            return out

        if isinstance(out, tuple):
            preds = out[0]
            aux = {} if len(out) < 2 else out[1]
            if not isinstance(aux, dict):
                aux = {"aux": aux}
            return ModelOutput(preds=preds, aux=aux)

        return ModelOutput(preds=out)

    def model_forward(
        self,
        lbw: Tensor,
        with_aux: bool = False,
    ) -> Union[Tensor, Dict[str, Tensor]]:
        _, _, C = lbw.shape

        x = lbw.clone()

        min_channels = self.target_channel_dim if self.local_norm_endo_only else C

        means = stdev = last_value = None

        if self.local_norm == "local_z":
            means = x.mean(1, keepdim=True).detach()
            var = x.var(1, keepdim=True, unbiased=False).detach()
            stdev = torch.sqrt(var) + 1e-5
            x[:, :, :min_channels] = (
                x[:, :, :min_channels] - means[:, :, :min_channels]
            ) / stdev[:, :, :min_channels]

        elif self.local_norm == "difference":
            last_value = x[:, -1:, :min_channels].detach().clone()
            x[:, :, :min_channels] = torch.diff(
                x[:, :, :min_channels], dim=1, prepend=x[:, :1, :min_channels]
            )

        out = self._call_model(x)
        pred = out.preds

        if self.local_norm == "local_z":
            pred[:, :, : self.target_channel_dim] = (
                pred[:, :, : self.target_channel_dim]
                * stdev[:, :1, : self.target_channel_dim]
                + means[:, :1, : self.target_channel_dim]
            )

        elif self.local_norm == "difference":
            pred[:, :, : self.target_channel_dim] = (
                torch.cumsum(pred[:, :, : self.target_channel_dim], dim=1)
                + last_value[:, :, : self.target_channel_dim]
            )

        return (pred, out.aux) if with_aux else pred

    def model_specific_forward(self, look_back_window: Tensor) -> Tensor:
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

        self.local_norm = datamodule.local_norm
        self.local_norm_endo_only = datamodule.local_norm_endo_only

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        if self.return_whole_series:
            series, lengths = batch
            loss = self.model_specific_train_step(series, lengths)
        else:
            look_back_window_norm, prediction_window_norm = batch
            loss = self.model_specific_train_step(
                look_back_window_norm, prediction_window_norm
            )
        return loss

    def validation_step(
        self, batch: Union[Tensor, Tuple[Tensor, Tensor]], batch_idx: int
    ) -> Tensor:
        if self.return_whole_series:
            series, lengths = batch
            loss = self.model_specific_val_step(series, lengths)
        else:
            look_back_window_norm, prediction_window_norm = batch
            loss = self.model_specific_val_step(
                look_back_window_norm, prediction_window_norm
            )
        return loss

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
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

    def _denormalize_tensor(
        self,
        preds: Tensor,
    ):
        device = preds.device
        if self.normalization == "global":
            train_dataset: Dataset = self.trainer.datamodule.train_dataset  # type:ignore
            mean = train_dataset.mean
            std = train_dataset.std
            mean = Tensor(mean).reshape(1, 1, -1).to(device).float()
            std = Tensor(std).reshape(1, 1, -1).to(device).float()
            preds = global_z_denorm(preds, mean, std)
        elif self.normalization == "minmax":
            train_dataset: Dataset = self.trainer.datamodule.train_dataset  # type:ignore
            min = train_dataset.min
            max = train_dataset.max
            min = Tensor(min).reshape(1, 1, -1).to(device).float()
            max = Tensor(max).reshape(1, 1, -1).to(device).float()
            preds = min_max_denorm(preds, min, max)

        return preds

    def evaluate(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int
    ) -> Tuple[dict[str, float], dict[str, list[float]]]:
        look_back_window_norm, prediction_window_norm = batch
        batch_size = look_back_window_norm.size(0)
        self.batch_size.append(batch_size)

        preds = self.model_forward(look_back_window_norm)
        preds = preds[:, :, : self.target_channel_dim]
        assert preds.shape == prediction_window_norm.shape

        # denormalize preds and prediction_window
        look_back_window = self._denormalize_tensor(look_back_window_norm)
        prediction_window = self._denormalize_tensor(prediction_window_norm)
        preds = self._denormalize_tensor(preds)
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
