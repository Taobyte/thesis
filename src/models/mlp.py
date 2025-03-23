import torch
import torch.nn as nn
import lightning as L

from typing import List
from torchvision.ops import MLP

from ..metrics import mae, mse, mape


class MLPBaseline(L.LightningModule):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        hidden_channels: List[int],
        dropout: float,
    ):
        super().__init__()

        self.model = MLP(
            look_back_window, hidden_channels + [prediction_window], dropout=dropout
        )
        self.loss = nn.L1Loss()

    def training_step(self, batch, batch_idx) -> float:
        x, y = batch
        preds = self.model(x)
        loss = self.loss(preds, y)
        return loss

    def validation_step(self, batch, batch_idx) -> None:
        x, y = batch
        preds = self.model(x)
        metric_mae = mae(preds, y)
        metric_mse = mse(preds, y)
        metric_mape = mape(preds, y)
        self.log("metric_mae", metric_mae)
        self.log("metric_mse", metric_mse)
        self.log("metric_mape", metric_mape)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
