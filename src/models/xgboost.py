import numpy as np
import torch
import wandb
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
        objective: str = "reg:squarederror",
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
        )


class XGBoost(BaseLightningModule):
    def __init__(
        self,
        loss: str = "MSE",
        target_channel_dim: int = 1,
        model: torch.nn.Module = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.target_channel_dim = target_channel_dim
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

    def log_tree_plots(self):
        import matplotlib.pyplot as plt

        num_trees_to_plot = 5
        for i in range(num_trees_to_plot):
            print(f"Plotting tree {i}")
            fig, ax = plt.subplots(figsize=(20, 12))

            xgb.plot_tree(
                self.model,
                num_trees=i,
                ax=ax,
                rankdir="LR",
            )
            ax.set_title(f"XGBoost Tree {i}", fontsize=16)
            plt.tight_layout()

            # Log the image to WandB using your Lightning logger's experiment
            self.logger.experiment.log(
                {
                    f"xgboost/tree_structure/tree_{i}": wandb.Image(
                        fig, caption=f"Tree {i}"
                    )
                },
            )

    # we use this hack!
    def on_train_epoch_start(self):
        self.model.fit(
            self.X_train,
            self.y_train,
            eval_set=[(self.X_val, self.y_val)],
            verbose=True,
        )
        # we need this for the optuna tuner
        preds = self.model.predict(self.X_val)
        preds = torch.tensor(preds, dtype=torch.float32, device=self.device)
        targets = torch.tensor(self.y_val, dtype=torch.float32, device=self.device)

        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, targets)
        else:
            loss = self.criterion(preds, targets)

        self.final_val_loss = loss

        self.log_feature_importance()
        # self.log_tree_plots()

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        self.log(
            "val_loss", self.final_val_loss, on_epoch=True, on_step=True, logger=True
        )

    def configure_optimizers(self):
        return None
