import torch

from torch import Tensor
from typing import Any

from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
        prediction_window: int,
        target_channel_dim: int = 1,
    ):
        super().__init__()
        self.prediction_window = prediction_window
        self.target_channel_dim = target_channel_dim

    def forward(self, x: Tensor) -> Tensor:
        B, _, _ = x.shape
        device = x.device
        return torch.zeros(
            (B, self.prediction_window, self.target_channel_dim), device=device
        )


class Dummy(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: str = "MSE",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model

    def model_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        return torch.Tensor([0])

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        return torch.Tensor([0])

    def configure_optimizers(self) -> None:
        return None
