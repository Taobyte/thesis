import numpy as np
import torch
import xgboost as xgb

from xgboost import XGBRegressor
from einops import rearrange

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class XGBoostModel(torch.nn.Module):
    def __init__(
        self,
        lbw_train_dataset: np.ndarray,
        pw_train_dataset: np.ndarray,
        lbw_val_dataset: np.ndarray,
        pw_val_dataset: np.ndarray,
        learning_rate: float = 0.001,
    ):
        super().__init__()

        self.lbw_train_dataset = lbw_train_dataset
        self.pw_train_dataset = pw_train_dataset
        self.lbw_val_dataset = lbw_val_dataset
        self.pw_val_dataset = pw_val_dataset

        self.model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=300,
            learning_rate=learning_rate,
            eval_metric="rmse",
            max_depth=4,
            reg_alpha=1,
            reg_lambda=1,
            subsample=0.8,
            colsample_bytree=0.8,
            # tree_method="gpu_hist",
            # predictor="gpu_predictor",
        )


class XGBoost(BaseLightningModule):
    def __init__(
        self,
        loss: str = "MSE",
        base_channel_dim: int = 1,
        model: torch.nn.Module = None,
    ):
        super().__init__()

        self.base_channel_dim = base_channel_dim
        self.model = model.model
        self.X_train = rearrange(model.lbw_train_dataset, "B T C -> B (T C)")
        self.y_train = rearrange(model.pw_train_dataset, "B T C -> B (T C)")
        self.X_val = rearrange(model.lbw_val_dataset, "B T C -> B (T C)")
        self.y_val = rearrange(model.pw_val_dataset, "B T C -> B (T C)")

        self.criterion = get_loss_fn(loss)
        self.automatic_optimization = False

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        look_back_window = look_back_window.detach().cpu().numpy()
        preds = self.model.predict(look_back_window)
        preds = rearrange(preds, "B (T C) -> B T C", C=self.base_channel_dim)
        # we need to have a pytorch tensor for evaluation
        preds = torch.tensor(preds, dtype=torch.float32)
        return preds

    def on_fit_end(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            # early_stopping_rounds=10,  # optional: stop if no improvement for 10 rounds
            # verbose=True,  # print eval results
        )

        # results = self.model.evals_results()

    # for now, these functions are not used, because we don't use batched training

    def model_specific_train_step(self, look_back_window, prediction_window):
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")
        self.model = xgb.train(
            self.xgboost_kwargs,
            dtrain=xgb.DMatrix(look_back_window, prediction_window),
            xgb_model=self.model,
        )
        preds = self.model.predict(xgb.DMatrix(look_back_window))
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        return None
