import torch
from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class Linear(BaseLightningModule):
    def __init__(
        self, model: torch.nn.Module, learning_rate: float = 0.01, loss: str = "MSE"
    ):
        super().__init__()
        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate

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
