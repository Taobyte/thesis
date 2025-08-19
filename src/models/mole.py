import torch
import torch.nn as nn

from einops import rearrange
from typing import Tuple, Any, Union

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()

        self.num_features = num_features
        self.eps = eps
        self.affine = affine

        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)

        elif mode == "denorm":
            x = self._denormalize(x)

        else:
            raise NotImplementedError

        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias

        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean

        return x


class HeadDropout(torch.nn.Module):
    def __init__(self, p=0.5):
        super(HeadDropout, self).__init__()
        if p < 0 or p > 1:
            raise ValueError(
                "Dropout probability has to be between 0 and 1, but got {}".format(p)
            )
        self.p = p

    def forward(self, x):
        # If in evaluation mode, return the input as-is
        if not self.training:
            return x

        # Create a binary mask of the same shape as x
        binary_mask = (torch.rand_like(x) > self.p).float()

        # Set dropped values to negative infinity during training
        return x * binary_mask + (1 - binary_mask) * -1e20


class Model(torch.nn.Module):
    def __init__(
        self,
        n_regimes: int = 2,
        dropout: float = 0.2,
        input_dropout: float = 0.0,
        look_back_window: int = 5,
        prediction_window: int = 3,
        look_back_channel_dim: int = 2,
        target_channel_dim: int = 1,
        use_last: bool = False,
        revin: bool = False,
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
                    prediction_window * look_back_channel_dim,
                )
                for _ in range(n_regimes)
            ]
        )
        if use_last:
            input_dim = look_back_channel_dim
        else:
            input_dim = look_back_window * look_back_channel_dim
        self.regime_layer = torch.nn.Sequential(
            torch.nn.Linear(input_dim, n_regimes),
            torch.nn.ReLU(),
            torch.nn.Linear(n_regimes, n_regimes),
        )
        self.head_dropout = HeadDropout(dropout)
        self.dropout = torch.nn.Dropout(input_dropout)

        self.use_last = use_last
        self.revin = revin
        self.rev = RevIN(look_back_channel_dim) if revin else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rev(x, "norm") if self.revin else self.rev(x)
        x = self.dropout(x)

        # flatten the input tensor
        x_reshaped = rearrange(x, "B L C -> B (L C)")
        predictions = torch.stack(
            [layer(x_reshaped) for layer in self.layers], dim=1
        )  # B n_regimes T*C
        mlp_input = (
            x_reshaped[:, -self.look_back_channel_dim :]
            if self.use_last
            else x_reshaped
        )
        mlp_out = self.regime_layer(mlp_input)  # (B n_regimes)
        head_dropout_out = self.head_dropout(mlp_out)
        weights = torch.nn.Softmax(dim=1)(head_dropout_out)  # (B n_regimes)
        weights = weights.unsqueeze(-1)  # (B n_regimes 1)
        weighted_pred = predictions * weights
        out = weighted_pred.sum(dim=1)
        out_reshaped = rearrange(out, "B (T C) -> B T C", C=self.look_back_channel_dim)

        out_reshaped_rev = (
            self.rev(out_reshaped, "denorm") if self.revin else self.rev(out_reshaped)
        )
        return out_reshaped_rev[:, :, : self.target_channel_dim]


class MOLE(BaseLightningModule):
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

    def configure_optimizers(
        self,
    ) -> Union[torch.optim.Adam, torch.optim.LBFGS, torch.optim.AdamW]:
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise NotImplementedError()

        return optimizer
