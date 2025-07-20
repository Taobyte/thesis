import torch

from einops import rearrange

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class Model(torch.nn.Module):
    def __init__(
        self,
        look_back_window: int = 5,
        prediction_window: int = 3,
        base_channel_dim: int = 1,
        input_channels: int = 1,
        hid_dim: int = 10,
        n_hid_layers: int = 2,
        activation: str = "relu",
        dropout: float = 0.0,
        use_norm: bool = False,
    ):
        super().__init__()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError(f"Unknown activation: {activation}")

        self.prediction_window = prediction_window
        self.base_channel_dim = base_channel_dim
        in_dim = look_back_window * input_channels
        out_dim = prediction_window * base_channel_dim

        self.layers = torch.nn.ModuleList()
        prev_dim = in_dim

        for _ in range(n_hid_layers):
            self.layers.append(torch.nn.Linear(prev_dim, hid_dim))
            self.layers.append(torch.nn.BatchNorm1d(hid_dim))
            self.layers.append(self.activation)
            self.layers.append(torch.nn.Dropout(dropout))
            prev_dim = hid_dim

        # Final layer without BatchNorm or Activation
        self.layers.append(torch.nn.Linear(prev_dim, out_dim))
        self.network = torch.nn.Sequential(*self.layers)

        self.use_norm = use_norm

    def forward(self, x: torch.Tensor):
        if self.use_norm:
            means = x.mean(1, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x /= stdev

        x = rearrange(x, "B T C -> B (T C)")
        pred = self.network(x)
        pred = rearrange(pred, "B (T C) -> B T C", C=self.base_channel_dim)

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
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.criterion = get_loss_fn(loss)

    def model_forward(self, look_back_window):
        return self.model(look_back_window)

    def model_specific_train_step(self, look_back_window, prediction_window):
        preds = self.model(look_back_window)
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds = self.model(look_back_window)
        loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
