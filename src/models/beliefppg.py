import torch
import numpy as np

from torch import Tensor
from typing import Any

from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return x


class BeliefPPG(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        prediction_window: int = 3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert self.normalization == "none"
        self.model = model
        self.prediction_window = prediction_window

    def on_train_epoch_start(self):
        data = self.trainer.datamodule.train_dataset.data
        logs = [np.log(h[1:, 0] / h[:-1, 0]) for h in data]

        concatenated = np.concatenate(logs)
        self.mean = torch.tensor(np.mean(concatenated))  # (1,)
        # self.std = np.std(concatenated)

    def model_forward(self, look_back_window: Tensor):
        B, _, _ = look_back_window.shape
        last_hr = look_back_window[:, -1, 0]
        preds = []

        for _ in range(self.prediction_window):
            mean_expanded = self.mean.expand(B)
            pred = torch.exp(last_hr.log() + mean_expanded)  # (B, 1)
            last_hr = pred
            preds.append(pred.unsqueeze(-1))
        concatenated = torch.concatenate(preds, dim=1).unsqueeze(-1)  # (B, T, 1)
        return concatenated

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        self.log("val_loss", 0, on_epoch=True)

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        pass

    def configure_optimizers(self) -> None:
        return None
