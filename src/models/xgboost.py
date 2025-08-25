import numpy as np
import torch
import wandb
import xgboost as xgb

from xgboost import XGBRegressor
from einops import rearrange
from numpy.typing import NDArray
from typing import Any, Tuple

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn

from src.normalization import local_z_norm_numpy


class XGBoostModel(torch.nn.Module):
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
        seed: int = 123,
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

        self.model = XGBRegressor(
            objective=objective,
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            eval_metric="rmse",
            max_depth=max_depth,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            # tree_method="gpu_hist",
            # predictor="gpu_predictor",
            callbacks=callbacks,
            random_state=seed,
        )


class XGBoost(BaseLightningModule):
    def __init__(
        self,
        model: XGBoostModel,
        loss: str = "MSE",
        verbose: bool = False,
        use_norm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_norm = use_norm

        self.model = model.model
        lbw_train = model.lbw_train_dataset
        pw_train = model.pw_train_dataset
        lbw_val = model.lbw_val_dataset
        pw_val = model.pw_val_dataset
        if use_norm:
            lbw_train, mean, std = local_z_norm_numpy(lbw_train)
            pw_train, _, _ = local_z_norm_numpy(pw_train, mean, std)
            lbw_val, mean, std = local_z_norm_numpy(lbw_val)
            pw_val, _, _ = local_z_norm_numpy(pw_val, mean, std)
        self.X_train = rearrange(lbw_train, "B T C -> B (T C)")
        self.y_train = rearrange(pw_train, "B T C -> B (T C)")
        self.X_val = rearrange(lbw_val, "B T C -> B (T C)")
        self.y_val = rearrange(pw_val, "B T C -> B (T C)")

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
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        look_back_window = look_back_window.detach().cpu().numpy()
        preds = self.model.predict(look_back_window)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        preds = rearrange(preds, "B (T C) -> B T C", C=self.target_channel_dim)
        # we need to have a pytorch tensor for evaluation
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        if self.use_norm:
            preds = preds * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )

            preds = preds + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )
        return preds

    def log_feature_importance(self):
        # Get feature importances
        booster = self.model.get_booster()
        importance = booster.get_score(importance_type="gain")  # or 'weight', 'cover'

        # Sort and prepare for logging
        importance = dict(
            sorted(importance.items(), key=lambda item: item[1], reverse=True)
        )

        table = wandb.Table(
            data=[[k, v] for k, v in importance.items()],
            columns=["Feature", "Importance"],
        )
        self.logger.experiment.log(
            {
                "XGBoost Feature Importance": wandb.plot.bar(
                    table, "Feature", "Importance", title="XGBoost Feature Importance"
                )
            }
        )

    # we use this hack!
    def on_train_epoch_start(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=self.verbose,
        )
        # we need this for the optuna tuner
        preds = self.model.predict(self.X_val)
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        targets = torch.tensor(self.y_val, dtype=torch.float32, device=self.device)

        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, targets)
        else:
            loss = self.criterion(preds, targets)

        self.final_val_loss = loss

        self.log_feature_importance()
        # self.log_tree_plots()

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
