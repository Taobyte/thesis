import torch

from einops import rearrange
from typing import Tuple, Any, Union

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
        n_regimes: int = 2,
        hidden_dim: int = 8,
        look_back_window: int = 5,
        prediction_window: int = 3,
        look_back_channel_dim: int = 2,
        target_channel_dim: int = 1,
    ):
        super().__init__()

        self.n_regimes = n_regimes
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.look_back_channel_dim = look_back_channel_dim
        self.target_channel_dim = target_channel_dim

        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    look_back_window * look_back_channel_dim,
                    prediction_window * target_channel_dim,
                )
                for _ in range(n_regimes)
            ]
        )
        self.regime_layer = torch.nn.Sequential(
            torch.nn.BatchNorm1d(look_back_window),
            torch.nn.Flatten(),
            torch.nn.Linear(look_back_window * look_back_channel_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, n_regimes),
            torch.nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # flatten the input tensor
        x_reshaped = rearrange(x, "B L C -> B (L C)")
        predictions = torch.stack(
            [layer(x_reshaped) for layer in self.layers], dim=1
        )  # B n_regimes T*C
        weights = self.regime_layer(x).unsqueeze(-1)  # B n_regimes 1
        weighted_pred = predictions * weights
        out = weighted_pred.sum(dim=1)
        out_reshaped = rearrange(out, "B (T C) -> B T C", C=self.target_channel_dim)
        return out_reshaped


class SETAR(BaseLightningModule):
    def __init__(
        self,
        model: Model,
        learning_rate: float = 0.001,
        loss_fn: str = "MSE",
        optimizer_name: str = "lbgfs",
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss_fn)
        self.mae_criterion = torch.nn.L1Loss()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.optimizer_name = optimizer_name

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        return self.model(look_back_window)

    def _shared_step(
        self, x: torch.Tensor, y: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        preds = self.model(x)
        loss = self.criterion(preds, y)
        mae_loss = self.mae_criterion(preds, y)

        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
        val_loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        loss = val_loss
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self) -> Union[torch.optim.Adam, torch.optim.LBFGS]:
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
        elif self.optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise NotImplementedError()

        return optimizer
