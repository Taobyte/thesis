import torch
import numpy as np
import pandas as pd
import wandb

from torch import Tensor
from typing import Any

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class Linear(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.01,
        loss: str = "MSE",
        use_scheduler: bool = False,
        weight_decay: float = 1e-4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler
        self.weight_decay = weight_decay

        self.mae_criterion = torch.nn.L1Loss()

    def model_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        preds = self.model(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        preds = self.model(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        if self.tune:
            loss = self.mae_criterion(preds, prediction_window)
        else:
            loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        return optimizer

    def on_fit_end(self):
        layer = self.model.layers[0]
        weights = layer.weight.detach().cpu().numpy()
        min_val = weights.min()
        max_val = weights.max()
        if max_val == min_val:
            normalized_weights = np.zeros_like(
                weights
            )  # Or all ones, depending on your preference
        else:
            normalized_weights = (weights - min_val) / (max_val - min_val)

        # heatmap
        self.logger.experiment.log(
            {
                "weights/linear_layer_weight_heatmap": wandb.Image(
                    normalized_weights,
                    caption="Linear Layer Heatmap (Min-Max Normalized)",
                )
            },
        )

        # weights histogram
        weights_flat = weights.flatten()
        self.logger.experiment.log(
            {"weights/linear_layer_weights_histogram": wandb.Histogram(weights_flat)},
        )
        # bias values
        if layer.bias is not None:
            biases = layer.bias.data.detach().cpu().numpy()
            df_bias = pd.DataFrame({"Index": np.arange(len(biases)), "Value": biases})
            self.logger.experiment.log(
                {"weights/linear_layer_bias_values": wandb.Table(dataframe=df_bias)},
                commit=False,
            )
        # weight values
        df_weights = pd.DataFrame(weights)
        self.logger.experiment.log(
            {"weights/linear_layer_weight_values": wandb.Table(dataframe=df_weights)}
        )
