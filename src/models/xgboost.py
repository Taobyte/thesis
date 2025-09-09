import numpy as np
import torch
import wandb
import xgboost as xgb

from xgboost import XGBRegressor
from einops import rearrange
from typing import Any, Tuple

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class XGBoostModel(torch.nn.Module):
    def __init__(
        self,
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
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.model = model.model
        self.verbose = verbose

        self.criterion = get_loss_fn(loss)
        self.automatic_optimization = False

    def model_specific_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        look_back_window = look_back_window.detach().cpu().numpy()
        preds = self.model.predict(look_back_window)
        if preds.ndim == 1:
            preds = preds[:, np.newaxis]
        preds = rearrange(preds, "B (T C) -> B T C", C=self.target_channel_dim)
        # we need to have a pytorch tensor for evaluation
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
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
        datamodule = self.trainer.datamodule
        lbw_train, pw_train, lbw_val, pw_val = datamodule.get_numpy_dataset()
        X_train = rearrange(lbw_train, "B T C -> B (T C)")
        y_train = rearrange(pw_train, "B T C -> B (T C)")
        X_val = rearrange(lbw_val, "B T C -> B (T C)")
        y_val = rearrange(pw_val, "B T C -> B (T C)")
        self.model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            verbose=self.verbose,
        )
        # we need this for the optuna tuner
        preds = self.model.predict(X_val)
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        if preds.ndim == 1:
            preds = preds.unsqueeze(-1)
        targets = torch.tensor(y_val, dtype=torch.float32, device=self.device)

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
