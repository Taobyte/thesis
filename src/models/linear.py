import torch

from torch import Tensor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from typing import Any
from einops import rearrange

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        target_channel_dim: int = 1,
        model_type: str = "linear_regression",
        alpha: float = 1.0,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.target_channel_dim = target_channel_dim
        if model_type == "ols":
            self.model = LinearRegression()
        elif model_type == "ridge":
            self.model = Ridge(alpha=alpha)
        elif model_type == "lasso":
            self.model = Lasso(alpha=alpha)
        elif model_type == "elastic_net":
            self.model = ElasticNet()

    def forward(self, look_back_window: Tensor) -> Tensor:
        device = look_back_window.device

        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        look_back_window = look_back_window.detach().cpu().numpy()
        pred = self.model.predict(look_back_window)
        pred = rearrange(pred, "B (T C) -> B T C", C=self.target_channel_dim)
        pred = torch.from_numpy(pred).to(device)

        return pred


class Linear(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: str = "MSE",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.automatic_optimization = False
        self.criterion = get_loss_fn(loss)
        self.mae_criterion = torch.nn.L1Loss()

    def model_specific_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        X_train, y_train, X_val, y_val = datamodule.get_numpy_dataset()

        X_train = rearrange(X_train, "B T C -> B (T C)")
        y_train = rearrange(y_train, "B T C -> B (T C)")

        self.model.model.fit(X_train, y_train)

        # we need this for the optuna tuner
        preds = self.model(torch.tensor(X_val))
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        targets = torch.tensor(y_val, dtype=torch.float32, device=self.device)

        if self.tune:
            loss = self.mae_criterion(preds, targets)
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
