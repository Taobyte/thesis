import torch

from torch import Tensor
from einops import rearrange
from typing import Any

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class Model(torch.nn.Module):
    def __init__(
        self,
        look_back_window: int = 5,
        prediction_window: int = 3,
        input_channels: int = 1,
        hid_dim: int = 10,
        n_hid_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        use_norm: bool = False,
        autoregressive: bool = False,
    ):
        super().__init__()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "gelu":
            self.activation = torch.nn.GELU()
        elif activation == "sigmoid":
            self.activation = torch.nn.Sigmoid()
        elif activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Unknown activation: {activation}")

        self.prediction_window = prediction_window
        self.input_channels = input_channels
        in_dim = look_back_window * input_channels
        out_dim = (
            input_channels if autoregressive else prediction_window * input_channels
        )

        self.layers = torch.nn.ModuleList()
        prev_dim = in_dim
        for _ in range(n_hid_layers):
            self.layers.append(torch.nn.Linear(prev_dim, hid_dim))
            self.layers.append(self.activation)
            self.layers.append(torch.nn.Dropout(dropout))
            prev_dim = hid_dim

        self.layers.append(torch.nn.Linear(prev_dim, out_dim))
        self.network = torch.nn.Sequential(*self.layers)

        self.use_norm = use_norm
        self.autoregressive = autoregressive

    def forward(self, x: Tensor) -> Tensor:
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = rearrange(x, "B T C -> B (T C)")
        if self.autoregressive:
            preds: list[Tensor] = []
            for _ in range(self.prediction_window):
                next_pred = self.network(x)
                preds.append(next_pred)
                x_unflattened = rearrange(x, "B (T C) -> B T C", C=self.input_channels)
                x = torch.concat(
                    (x_unflattened[:, 1:, :], next_pred.unsqueeze(1)), dim=1
                )
                x = rearrange(x, "B T C -> B (T C)")

            pred = torch.concat(preds, dim=1)
        else:
            pred = self.network(x)

        pred = rearrange(pred, "B (T C) -> B T C", C=self.input_channels)

        if self.use_norm:
            pred = pred * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )

            pred = pred + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )
        return pred


class MLP(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.001,
        loss: str = "MSE",
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = get_loss_fn(loss)
        self.mae_loss = torch.nn.L1Loss()

    def model_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        assert preds.shape == prediction_window.shape
        if self.tune:
            loss = self.mae_loss(preds, prediction_window)
        else:
            loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
