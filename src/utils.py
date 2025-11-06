import os
import yaml
import numpy as np
import lightning as L

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint, Callback
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning import LightningDataModule, Trainer, LightningModule
from typing import Tuple, Union, List

from src.models.utils import get_model_kwargs
from src.callbacks import EfficiencyCallback, PredictionCallback


def delete_checkpoint(trainer: Trainer, checkpoint_callback: ModelCheckpoint):
    if trainer.is_global_zero:
        best_checkpoint_path: str = checkpoint_callback.best_model_path  # ignore:type
        if best_checkpoint_path and os.path.isfile(best_checkpoint_path):
            os.remove(best_checkpoint_path)
            print(f"Successfully deleted best checkpoint: {best_checkpoint_path}")


def setup(
    config: DictConfig,
    wandb_logger: WandbLogger,
    run_name: str,
    fitted_models: list[LightningModule] = [],
) -> Tuple[LightningDataModule, LightningModule, Trainer, list[Callback]]:
    return_whole_series = False
    if config.model.name in config.return_series_models:
        return_whole_series = True

    datamodule = instantiate(
        config.dataset.datamodule,
        return_whole_series=return_whole_series,
    )

    model_kwargs = get_model_kwargs(config, datamodule)
    if fitted_models:  # needed for ensemble model
        model_kwargs["fitted_models"] = fitted_models
    model = instantiate(config.model.model, **model_kwargs)
    pl_model = instantiate(
        config.model.pl_model, model=model, return_whole_series=return_whole_series
    )

    callbacks: List[Union[Callback, EarlyStopping, ModelCheckpoint]] = []

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename=run_name + "-{epoch}-{step}",
        save_top_k=1,
    )
    callbacks.append(checkpoint_callback)

    if config.model.trainer.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.model.trainer.patience,
                min_delta=config.min_delta,  # stop if no substantial improvement is being made. Important for models with LR schedulers
            )
        )

    if config.use_efficiency_callback:
        efficiency_callback = EfficiencyCallback()
        callbacks.append(efficiency_callback)

    if config.use_prediction_callback:
        prediction_callback = PredictionCallback()
        callbacks.append(prediction_callback)

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.model.trainer.max_epochs,
        callbacks=callbacks,
        enable_model_summary=False,
        overfit_batches=1 if config.overfit else 0.0,
        limit_test_batches=10 if config.overfit else None,
        default_root_dir=config.path.checkpoint_path,
        num_sanity_val_steps=0,
        enable_progress_bar=config.progress_bar,
    )

    return datamodule, pl_model, trainer, callbacks


def create_group_run_name(
    dataset_name: str,
    model_name: str,
    look_back_window: int,
    prediction_window: int,
    fold_nr: int = 0,
    normalization: str = "global",
    seed: int = 0,
    local_norm: str = "lnone",
    local_norm_endo_only: bool = False,
    feature_name: str = "mean",
) -> Tuple[str, str, list[str]]:
    fold = f"f{fold_nr}"

    ln_endo_only_tag = "ln_eo" if local_norm_endo_only else "ln_ee"
    group_name = (
        f"{normalization}-{local_norm}-{ln_endo_only_tag}"
        f"-{dataset_name}-imu_{feature_name}"
    )

    tags: list[str] = [dataset_name, model_name, normalization]
    if dataset_name in ["ldalia", "lwildppg", "lieee", "limitbih"]:
        # local datasets do not use folds
        run_name = (
            group_name
            + f"-{model_name}-lb{look_back_window}"
            + f"-pw{prediction_window}-s{seed}"
        )

    else:
        run_name = (
            group_name
            + f"-{model_name}-lb{look_back_window}"
            + f"-pw{prediction_window}-{fold}-s{seed}"
        )
        tags.append(fold)

    if local_norm in ["local_z", "difference"]:
        tags.append(local_norm)
        tags.append(ln_endo_only_tag)

    if feature_name != "none":
        tags.append(feature_name)

    return group_name, run_name, tags


def get_optuna_name(
    dataset_name: str,
    use_heart_rate: bool,
    look_back_window: int,
    prediction_window: int,
    fold_nr: int = 0,
    fold_datasets: list[str] = ["dalia", "wildppg", "ieee"],
    normalization: str = "global",
):
    group_name, _, tags = create_group_run_name(
        dataset_name,
        model_name,
        look_back_window,
        prediction_window,
        fold_nr,
        fold_datasets,
        normalization,
    )
    # tags[1] stores the models name
    return f"optuna_{tags[1]}_{group_name}"


def setup_wandb_logger(
    config: DictConfig,
) -> Tuple[Union[WandbLogger, DummyLogger], str]:
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    group_name, run_name, tags = create_group_run_name(
        config.dataset.name,
        config.model.name,
        config.look_back_window,
        config.prediction_window,
        config.folds.fold_nr,
        config.normalization,
        seed=config.seed,
        local_norm=config.local_norm,
        local_norm_endo_only=config.local_norm_endo_only,
        feature_name=config.feature.name,
    )

    # print(config_dict)
    if config.tune:
        config.use_wandb = False
        return DummyLogger(), run_name

    wandb_logger = (
        WandbLogger(
            name=run_name,
            group=group_name,
            config=config_dict,
            tags=tags,
            **config.wandb,
        )
        if config.use_wandb
        else DummyLogger()
    )
    print("Setup complete.")
    print(f"Run name: {run_name}")
    print(f"Group name: {group_name}")
    print(f"Tags: {tags}")

    return wandb_logger, run_name


# -------------------------------------------------------------------------------------------------------
# OmegaConf custom resolver functions
# -------------------------------------------------------------------------------------------------------


def get_min(x: str, y: str) -> int:
    return min(int(x), int(y))


def resolve_str(x: int) -> str:
    return str(x)


# this function is needed, because AdaMSHyper does not support short look_back_window lengths
# the custom resolver function is used in the model definition config/model/adamshyper.yaml
def compute_square_window(seq_len: int, max_window: int = 4) -> list[int]:
    """
    Finds the largest equal factors [k,k] such that k*k <= seq_len.
    Defaults to [4,4] if to valid window exists.
    """
    max_k = int(np.sqrt(seq_len))
    max_k = min(max_k, max_window)
    return [max_k, max_k]


def number_of_exo_vars(features: list[str]) -> int:
    n_catch22 = int(np.sum(["catch22" in f for f in features]))
    out = len(features) - n_catch22 + 24 * n_catch22
    return out


def compute_input_channel_dims(
    target_channel_dim: int,
    dynamic_exogenous_variables: int,
    static_exogenous_variables: int,
    use_dynamic_features: bool,
    use_static_features: bool,
) -> int:
    dims = target_channel_dim
    if use_dynamic_features:
        dims += dynamic_exogenous_variables
    if use_static_features:
        dims += static_exogenous_variables

    return dims
