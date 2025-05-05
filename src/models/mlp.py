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
    ):
        super().__init__()

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError()

        self.base_channel_dim = base_channel_dim

        in_dim = look_back_window * input_channels
        out_dim = prediction_window * base_channel_dim

        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.layer_sizes[i - 1], self.layer_sizes[i])
                for i in range(1, len(self.layer_sizes))
            ]
        )

    def forward(self, x: torch.Tensor):
        x = rearrange(x, "B T C -> B (T C)")
        pred = self.layers(x)
        pred = rearrange(pred, "B (T C) -> B T C", C=self.base_channel_dim)
        return pred


class MLP(BaseLightningModule):
    def __init__(
        self, model: torch.nn.Module, learning_rate: float = 0.001, loss: str = "MSE"
    ):
        super().__init__()
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
