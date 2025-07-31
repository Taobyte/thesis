import torch
import numpy as np

from torch import Tensor
from sklearn.linear_model import LinearRegression
from numpy.typing import NDArray
from typing import Any


from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule
from src.models.mlp import Model


class HLinearModel(torch.nn.Module):
    def __init__(
        self,
        lbw_train_dataset: NDArray[np.float32],
        pw_train_dataset: NDArray[np.float32],
        lbw_val_dataset: NDArray[np.float32],
        pw_val_dataset: NDArray[np.float32],
        look_back_window: int,
        prediction_window: int,
        target_channel_dim: int = 1,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.target_channel_dim = target_channel_dim

        self.endo_model = LinearRegression()
        self.exo_model = LinearRegression()

        self.lbw_train_dataset = lbw_train_dataset
        self.pw_train_dataset = pw_train_dataset
        self.lbw_val_dataset = lbw_val_dataset
        self.pw_val_dataset = pw_val_dataset

    def forward(self, x: Tensor) -> Tensor:
        _, _, c = x.shape
        device = x.device
        assert c > 1
        endo = x[:, :, : self.target_channel_dim]
        exo = x[:, :, self.target_channel_dim :]
        endo = endo.squeeze(-1).detach().cpu().numpy()  # (B, L)
        exo = exo.squeeze(-1).detach().cpu().numpy()  # (B, L)
        initial_pred = self.endo_model.predict(endo)
        deltas = self.exo_model.predict(exo)
        pred = initial_pred + deltas
        pred = torch.from_numpy(pred).to(device).unsqueeze(-1)
        return pred


class HLinear(BaseLightningModule):
    def __init__(
        self, model: Model, learning_rate: float, loss: str = "MSE", **kwargs: Any
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = get_loss_fn(loss)
        assert self.experiment_name == "endo_exo"

        self.automatic_optimization = False

    def model_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    # we use this hack!
    def on_train_epoch_start(self):
        X_train = self.model.lbw_train_dataset
        y_train = self.model.pw_train_dataset
        X_val = self.model.lbw_val_dataset
        y_val = self.model.pw_val_dataset

        endo_X_train = X_train[:, :, 0]
        endo_y_train = y_train[:, :, 0]
        exo_X_train = X_train[:, :, 1]
        # exo_y_train = y_val[:, :, 1]

        self.model.endo_model.fit(endo_X_train, endo_y_train)
        residuals = endo_y_train - self.model.endo_model.predict(endo_X_train)
        self.model.exo_model.fit(exo_X_train, residuals)

        # we need this for the optuna tuner
        preds = self.model(torch.tensor(X_val))
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        targets = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, targets)
        else:
            loss = self.criterion(preds, targets)

        self.final_val_loss = loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        return torch.Tensor([0])

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        self.log(
            "val_loss", self.final_val_loss, on_epoch=True, on_step=True, logger=True
        )
        return torch.Tensor([0])

    def configure_optimizers(self) -> None:
        return None
