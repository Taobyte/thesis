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
from src.normalization import (
    local_z_denorm,
    local_z_norm,
    global_z_denorm,
    global_z_norm,
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


class BaseLightningModule(L.LightningModule):
    def __init__(
        self,
        wandb_project: str = "c_keusch/thesis",
        n_trials: int = 10,
        name: str = None,
        use_plots: bool = False,
        normalization: str = "local",
        tune: bool = False,
        use_only_exogenous_features: bool = False,
    ):
        super().__init__()

        self.n_trials = n_trials
        self.wandb_project = wandb_project
        self.name = name
        self.tune = tune
        self.use_plots = use_plots
        self.normalization = normalization
        self.use_only_exogenous_features = use_only_exogenous_features

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
        return self.name in ["gp", "dklgp", "exactgp", "bnn", "dropoutbnn"]

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

    def on_train_end(self):
        # self.compute_shap_values()
        pass

    def _normalize_data(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        if self.normalization == "local":
            look_back_window_norm, mean, std = local_z_norm(
                look_back_window, self.local_norm_channels
            )
            prediction_window_norm, _, _ = local_z_norm(
                prediction_window, self.local_norm_channels, mean, std
            )
            return look_back_window_norm, prediction_window_norm, mean, std
        elif self.normalization == "global":
            datamodule = self.trainer.datamodule
            mean = datamodule.train_dataset.mean
            std = datamodule.train_dataset.std

            look_back_window_norm = global_z_norm(
                look_back_window, self.local_norm_channels, mean, std
            )
            prediction_window_norm = global_z_norm(
                prediction_window, self.local_norm_channels, mean, std
            )
            return look_back_window_norm, prediction_window_norm
        else:
            raise NotImplementedError()

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        look_back_window, prediction_window = batch

        # normalize data
        if self.normalization == "global":
            look_back_window_norm, prediction_window_norm = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "local":
            look_back_window_norm, prediction_window_norm, _, _ = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "none":
            look_back_window_norm, prediction_window_norm = batch

        if self.use_only_exogenous_features:
            assert self.use_dynamic_features or self.use_static_features, (
                "Attention! You train with only exogenous variables, but don't use dynamic or static exogenous features"
            )
            look_back_window_norm = look_back_window_norm[
                :, :, self.target_channel_dim :
            ]

        loss = self.model_specific_train_step(
            look_back_window_norm, prediction_window_norm
        )
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx):
        look_back_window, prediction_window = batch

        # normalize data
        if self.normalization == "global":
            look_back_window_norm, prediction_window_norm = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "local":
            look_back_window_norm, prediction_window_norm, _, _ = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "none":
            look_back_window_norm, prediction_window_norm = batch

        if self.use_only_exogenous_features:
            assert self.use_dynamic_features or self.use_static_features, (
                "Attention! You train with only exogenous variables, but don't use dynamic or static exogenous features"
            )
            look_back_window_norm = look_back_window_norm[
                :, :, self.target_channel_dim :
            ]

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

        if self.normalization == "global":
            look_back_window_norm, _ = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "local":
            look_back_window_norm, _, mean, std = self._normalize_data(
                look_back_window, prediction_window
            )
        elif self.normalization == "none":
            look_back_window_norm = look_back_window
        else:
            raise NotImplementedError()

        if self.use_only_exogenous_features:
            assert self.use_dynamic_features or self.use_static_features, (
                "Attention! You train with only exogenous variables, but don't use dynamic or static exogenous features"
            )
            look_back_window_norm = look_back_window_norm[
                :, :, self.target_channel_dim :
            ]
            look_back_window = look_back_window[:, :, self.target_channel_dim :]

        # Prediction
        if self.has_probabilistic_forecast:
            preds, _ = self.model_forward(look_back_window_norm)
        else:
            preds = self.model_forward(look_back_window_norm)

        preds = preds[:, :, : prediction_window.shape[-1]]

        assert preds.shape == prediction_window.shape
        if self.normalization == "local":
            denormalized_preds = local_z_denorm(
                preds, self.local_norm_channels, mean, std
            )
        elif self.normalization == "global":
            datamodule = self.trainer.datamodule
            mean = datamodule.train_dataset.mean
            std = datamodule.train_dataset.std
            denormalized_preds = global_z_denorm(
                preds, self.local_norm_channels, mean, std
            )
        elif self.normalization == "none":
            denormalized_preds = preds
        else:
            raise NotImplementedError()

        # Metric Calculation
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

    def compute_shap_values(self):
        assert self.normalization != "local"

        import shap
        from einops import rearrange

        class ModelWrapper(torch.nn.Module):
            def __init__(self, model, look_back_channel_dim, target_channel_dim):
                super().__init__()
                self.model = model
                self.look_back_channel_dim = look_back_channel_dim
                self.target_channel_dim = target_channel_dim

            def forward(self, x):
                x = rearrange(x, "B (T C) -> B T C", C=self.look_back_channel_dim)
                output = self.model(x)
                output = output[:, :, : self.target_channel_dim]
                output = rearrange(output, "B T C -> B (T C)")
                return output

        # self.model.eval()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_to_explain = ModelWrapper(
            self.model,
            self.trainer.datamodule.look_back_channel_dim,
            self.trainer.datamodule.target_channel_dim,
        )

        background_tensor = self.trainer.datamodule.get_inducing_points(
            strategy="kmeans", mode="train"
        )

        _, look_back_window, channels = background_tensor.shape

        background_tensor = background_tensor.to(device)
        background_tensor = rearrange(background_tensor, "B T C -> B (T C)")
        background_tensor.requires_grad_()
        explain_tensor = self.trainer.datamodule.get_inducing_points(
            strategy="kmeans", mode="test", num_inducing=50
        )
        explain_tensor = explain_tensor.to(device).requires_grad_(True)
        explain_tensor = rearrange(explain_tensor, "B T C -> B (T C)")
        explain_tensor.requires_grad_()

        explainer = shap.DeepExplainer(model_to_explain, background_tensor)

        shap_values = explainer.shap_values(explain_tensor)

        feature_names = []
        for i in range(look_back_window):
            for j in range(channels):
                feature_names.append(f"Activity {i + 1}" if j == 1 else f"HR {i + 1}")

        explanation = shap.Explanation(
            values=shap_values[:, :, 0],  # shape (B, T*C)
            data=explain_tensor.detach().cpu().numpy(),
            feature_names=feature_names,
        )

        shap.plots.beeswarm(explanation)

        fst_shap = shap_values[0]
        fst_shap = rearrange(fst_shap, "(T C) L -> T C L", C=self.look_back_channel_dim)
        _, _, L = fst_shap.shape
        import matplotlib.pyplot as plt
        import seaborn as sns

        for t in range(L):
            sns.heatmap(fst_shap[:, :, t], annot=True, cmap="viridis", cbar=True)
            plt.title("5x2 Heatmap")
            plt.xlabel("Column")
            plt.ylabel("Row")
            plt.show()
