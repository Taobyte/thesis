import numpy as np
import torch
import xgboost as xgb

from xgboost import XGBRegressor
from einops import rearrange
from numpy.typing import NDArray
from typing import Any, Tuple

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn

from src.normalization import local_z_norm_numpy


class HXGBoostModel(torch.nn.Module):
    def __init__(
        self,
        lbw_train_dataset: NDArray[np.float32],
        pw_train_dataset: NDArray[np.float32],
        lbw_val_dataset: NDArray[np.float32],
        pw_val_dataset: NDArray[np.float32],
        learning_rate: float = 0.001,
        loss: str = "MSE",
        n_estimators: int = 300,
        max_depth: int = 4,
        reg_alpha: int = 1,
        reg_lambda: int = 1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        use_early_stopping: bool = False,
        patience: int = 2,
    ):
        super().__init__()

        self.lbw_train_dataset = lbw_train_dataset
        self.pw_train_dataset = pw_train_dataset
        self.lbw_val_dataset = lbw_val_dataset
        self.pw_val_dataset = pw_val_dataset

        if loss == "MSE":
            objective = "reg:squarederror"
        elif loss == "MAE":
            objective = "reg:absoluteerror"
        else:
            raise NotImplementedError()

        callbacks = []
        if use_early_stopping:
            early_stop = xgb.callback.EarlyStopping(
                rounds=patience,
                metric_name="rmse",
                data_name="validation_0",
                save_best=True,
                maximize=False,
            )
            callbacks.append(early_stop)

        self.endo_model = XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="rmse",
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            callbacks=callbacks,
        )

        self.exo_model = XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="rmse",
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            callbacks=callbacks,
        )


class HXGBoost(BaseLightningModule):
    def __init__(
        self,
        model: HXGBoostModel,
        loss: str = "MSE",
        verbose: bool = False,
        use_norm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        assert self.experiment_name == "endo_exo"

        self.use_norm = use_norm

        self.endo_model = model.endo_model
        self.exo_model = model.exo_model
        lbw_train = model.lbw_train_dataset
        pw_train = model.pw_train_dataset
        lbw_val = model.lbw_val_dataset
        pw_val = model.pw_val_dataset
        if use_norm:
            lbw_train, mean, std = local_z_norm_numpy(lbw_train)
            pw_train = local_z_norm_numpy(pw_train, mean, std)
            lbw_val, mean, std = local_z_norm_numpy(lbw_val)
            pw_val = local_z_norm_numpy(pw_val, mean, std)

        self.X_train = lbw_train
        self.X_val = lbw_val
        self.y_train = pw_train
        self.y_val = pw_val

        self.criterion = get_loss_fn(loss)
        self.automatic_optimization = False

        self.verbose = verbose

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        if self.use_norm:
            means = look_back_window.mean(1, keepdim=True).detach()
            look_back_window = look_back_window - means
            stdev = torch.sqrt(
                torch.var(look_back_window, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            look_back_window /= stdev

        endo_lbw = look_back_window[:, :, 0].detach().cpu().numpy()
        exo_lbw = look_back_window[:, :, 1].detach().cpu().numpy()
        initial_preds = self.endo_model.predict(endo_lbw)
        deltas = self.exo_model.predict(exo_lbw)
        preds = initial_preds + deltas
        preds = rearrange(preds, "B (T C) -> B T C", C=self.target_channel_dim)
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        if self.use_norm:
            preds = preds * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )

            preds = preds + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )
        return preds

    # we use this hack!
    def on_train_epoch_start(self):
        endo_X_train = self.X_train[:, :, 0]
        endo_y_train = self.y_train[:, :, 0]
        endo_X_val = self.X_val[:, :, 0]
        endo_y_val = self.y_val[:, :, 0]

        self.endo_model.fit(
            endo_X_train,
            endo_y_train,
            eval_set=[(endo_X_val, endo_y_val)],
            verbose=self.verbose,
        )

        initial_preds_train = self.endo_model.predict(endo_X_train)
        train_residuals = endo_y_train - initial_preds_train

        initial_preds_val = self.endo_model.predict(endo_X_val)
        val_residuals = endo_y_val - initial_preds_val
        exo_X_train = self.X_train[:, :, 1]
        exo_X_val = self.X_val[:, :, 1]

        self.exo_model.fit(
            exo_X_train,
            train_residuals,
            eval_set=[(exo_X_val, val_residuals)],
            verbose=self.verbose,
        )

        # we need this for the optuna tuner
        init_preds = self.endo_model.predict(endo_X_val)
        deltas = self.exo_model.predict(exo_X_val)
        preds = init_preds + deltas
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)  # (B ,T)
        preds = preds.unsqueeze(-1)  # (B, T, 1)
        targets = torch.tensor(
            endo_y_val, dtype=torch.float32, device=self.device
        )  # (B, T)
        targets = targets.unsqueeze(-1)  # (B, T, 1)

        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, targets)
        else:
            loss = self.criterion(preds, targets)

        self.final_val_loss = loss

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return torch.Tensor()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        self.log(
            "val_loss", self.final_val_loss, on_epoch=True, on_step=True, logger=True
        )

    def configure_optimizers(self):
        return None
