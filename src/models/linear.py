import torch
from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class Linear(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.01,
        loss: str = "MSE",
        use_scheduler: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

    def model_forward(self, look_back_window):
        return self.model(look_back_window)

    def model_specific_train_step(self, look_back_window, prediction_window):
        preds = self.model(look_back_window)
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds = self.model(look_back_window)
        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, prediction_window)
        else:
            loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        if self.use_scheduler:
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, 1, gamma=0.1, last_epoch=-1
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "name": "StepLR",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return optimizer
