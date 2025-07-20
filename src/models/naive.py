import torch
from src.models.utils import BaseLightningModule


class DummyModel(torch.nn.Module):
    def __init__(self, use_trend: bool = False, prediction_window: int = 3):
        super().__init__()
        self.use_trend = use_trend
        self.prediction_window = prediction_window

    def forward(self, look_back_window: torch.Tensor):
        B, L, C = look_back_window.shape

        last_value = (
            look_back_window[:, -1, 0].unsqueeze(1).expand(B, self.prediction_window)
        )
        if self.use_trend:
            slope = (look_back_window[:, -1, 0] - look_back_window[:, -2, 0]).unsqueeze(
                1
            ) * torch.arange(1, self.prediction_window + 1).repeat(B, 1)
        else:
            slope = torch.zeros(B, self.prediction_window)
        pred = last_value + slope
        return pred.unsqueeze(-1)


class Naive(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        use_trend: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.use_trend = use_trend
        self.model = model

    def model_forward(self, look_back_window):
        preds = self.model(look_back_window)
        return preds

    def model_specific_train_step(self, look_back_window, prediction_window) -> float:
        last_value = self.model_forward(look_back_window)
        loss = torch.nn.MSELoss(last_value, prediction_window)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window) -> float:
        last_value = self.model_forward(look_back_window)
        loss = torch.nn.MSELoss(last_value, prediction_window)
        return loss

    def configure_optimizers(self):
        return None
