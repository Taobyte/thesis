import torch

from einops import rearrange

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
        look_back_window: int = 5,
        prediction_window: int = 3,
        base_channel_dim: int = 1,
        input_channels: int = 1,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.base_channel_dim = base_channel_dim
        self.input_channels = input_channels

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(input_channels * (look_back_window + i), input_channels)
                for i in range(prediction_window)
            ]
        )  # these are the transition matrices from the KalmanFilter for each timestep

    def forward(self, x: torch.Tensor):
        assert len(x.shape) == 2

        for layer in self.layers:
            x_hat = layer(x)
            x = torch.cat((x, x_hat), dim=1)

        pred = x[:, -(self.prediction_window * self.base_channel_dim) :]
        return pred


class KalmanFilter(BaseLightningModule):
    def __init__(
        self, model: torch.nn.Module, loss: str = "MSE", learning_rate: float = 0.001
    ):
        super().__init__()

        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate

    def model_forward(self, look_back_window):
        assert len(look_back_window.shape) == 3

        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        preds = self.model(look_back_window)
        preds = rearrange(preds, "B (T C) -> B T C", C=self.model.base_channel_dim)

        return preds

    def model_specific_train_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)
        loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
