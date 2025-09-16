import torch
import torch.nn as nn

from torch import Tensor
from lightning import LightningModule
from typing import Any, Tuple
from einops import rearrange

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class Model(nn.Module):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        hidden_dim: int = 16,
        target_channel_dim: int = 1,
        look_back_channel_dim: int = 2,
        fitted_models: list[LightningModule] = [],
        freeze_experts: bool = True,
        softmax_temp: float = 1.0,
        use_mixture_net: bool = True,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.target_channel_dim = target_channel_dim

        self.models = nn.ModuleList(fitted_models)
        self.n_models = len(fitted_models)

        if freeze_experts:
            for m in self.models:
                for p in m.parameters():
                    p.requires_grad = False
                m.eval()  # disable dropout/bn updates

        self.use_mixture_net = use_mixture_net
        if self.use_mixture_net:
            self.softmax_temp = softmax_temp
            self.mixture_net = nn.Sequential(
                nn.Linear(look_back_channel_dim * look_back_window, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.n_models),
                nn.Softmax(dim=-1),
            )
        else:
            self.weights = torch.nn.Parameter(
                torch.randn(
                    self.n_models,
                )
            )

    def forward(self, look_back_window: Tensor) -> Tensor:
        x = rearrange(look_back_window, "B T C -> B (T C)")
        if self.use_mixture_net:
            if self.softmax_temp != 1.0:
                h = self.mixture_net[0](x)
                h = self.mixture_net[1](h)
                logits = self.mixture_net[2](h) / self.softmax_temp
                weights = torch.softmax(logits, dim=-1)
            else:
                weights = self.mixture_net(x)
        else:
            B, _, _ = look_back_window.shape
            weights = torch.nn.functional.softmax(self.weights, dim=-1)
            weights = weights.unsqueeze(0).expand(B, -1)

        preds = []
        for m in self.models:
            p = m.model_forward(look_back_window)
            preds.append(p)

        P = torch.stack(preds, dim=-1)

        w = weights.unsqueeze(1).unsqueeze(1)
        pred = (P * w).sum(dim=-1)

        return pred


class Ensemble(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: str = "MSE",
        learning_rate: float = 0.001,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss)
        self.mae_criterion = torch.nn.L1Loss()

        self.learning_rate = learning_rate

    def model_specific_forward(self, look_back_window: Tensor):
        return self.model(look_back_window)

    def _shared_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor]:
        preds = self.model_forward(look_back_window)
        loss = self.criterion(preds, prediction_window)
        mae_loss = self.mae_criterion(preds, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            loss = mae_loss

        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Adam:
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
