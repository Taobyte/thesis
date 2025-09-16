import torch
import numpy as np
import pandas as pd
import wandb
import gc

from lightning import LightningModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from typing import Tuple, List, Optional
from collections import defaultdict

from hydra import compose
from hydra.core.hydra_config import HydraConfig

from src.utils import setup, delete_checkpoint


def tune(config: DictConfig, wandb_logger: WandbLogger, run_name: str) -> float:
    dataset_name = config.dataset.name

    # loop over all folds and return average val loss performance
    val_losses: list[float] = []
    for i in range(config.n_folds):
        if dataset_name in config.fold_datasets:
            base_path = get_original_cwd()
            fold_name = f"fold_{i}"
            fold_conf_path = f"{base_path}/config/folds/{fold_name}.yaml"
            fold_config = OmegaConf.load(fold_conf_path)

            config["folds"][dataset_name]["train_participants"] = fold_config[
                dataset_name
            ]["train_participants"]

            config["folds"][dataset_name]["val_participants"] = fold_config[
                dataset_name
            ]["val_participants"]

        else:
            config["seed"] = i

        datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
        checkpoint_callback = callbacks[0]

        print(f"Starting fold {i}")
        trainer.fit(pl_model, datamodule=datamodule)
        if config.model.name not in config.special_models:
            val_results = trainer.validate(
                datamodule=datamodule, ckpt_path="best"
            )  # use best model to validate
        else:
            val_results = trainer.validate(pl_model, datamodule=datamodule)
        last_val_loss = val_results[0]["val_loss_epoch"]
        val_losses.append(last_val_loss)
        print(f"Finished fold {i}, val_loss = {last_val_loss:.4f}")

        delete_checkpoint(trainer, checkpoint_callback)

        del datamodule, pl_model, trainer, callbacks
        gc.collect()
        torch.cuda.empty_cache()

    avg_val_loss = float(np.mean(val_losses))
    print(f"Average validation loss across folds: {avg_val_loss:.4f}")
    return avg_val_loss


def tune_local(config: DictConfig, wandb_logger: WandbLogger, run_name: str) -> float:
    results: List[float] = []
    for participant in config.dataset.participants:
        datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
        datamodule.participant = participant
        checkpoint_callback = callbacks[0]
        trainer.fit(pl_model, datamodule=datamodule)
        if config.model.name not in config.special_models:
            val_results = trainer.validate(datamodule=datamodule, ckpt_path="best")
        else:
            val_results = trainer.validate(pl_model, datamodule=datamodule)
        results.append(val_results[0]["val_loss_epoch"])
        delete_checkpoint(trainer, checkpoint_callback)
        del datamodule, pl_model, trainer, callbacks
        gc.collect()
        torch.cuda.empty_cache()

    averaged_results = float(np.mean(results))
    print(f"Average validation loss across folds: {averaged_results:.4f}")
    return averaged_results


def train_test_global(
    config: DictConfig, wandb_logger: WandbLogger, run_name: str
) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
    datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
    checkpoint_callback = callbacks[0]

    print("Start Training.")
    trainer.fit(pl_model, datamodule=datamodule)
    print("End Training.")

    print("Start Evaluation.")
    if config.model.name not in config.special_models:
        global_test_results = trainer.test(
            pl_model, datamodule=datamodule, ckpt_path="best"
        )
    else:
        print("Best checkpoint not found, testing with current model.")
        global_test_results = trainer.test(
            pl_model, datamodule=datamodule, ckpt_path=None
        )

    del datamodule
    if config.test_local:
        ckpt_path = checkpoint_callback.best_model_path or None
        dummy_logger = DummyLogger()
        local_datamodule, _, local_trainer, _ = setup(config, dummy_logger, run_name)
        local_datamodule.test_local = True
        local_datamodule.setup("fit")  # initialize train_dataset for normalization

        if ckpt_path and (config.model.name not in config.special_models):
            local_test_results = local_trainer.test(
                pl_model, datamodule=local_datamodule, ckpt_path=ckpt_path
            )
        else:
            print("Best checkpoint not found, testing with current model.")
            local_test_results = local_trainer.test(
                pl_model, datamodule=local_datamodule, ckpt_path=None
            )

        delete_checkpoint(trainer, checkpoint_callback)

        print("End Evaluation.")

        global_results = pd.DataFrame(global_test_results)
        local_results = pd.DataFrame(local_test_results)

        wandb_logger.experiment.log(
            {
                "local_results": wandb.Table(dataframe=local_results),
            }
        )

        return global_results, local_results

    delete_checkpoint(trainer, checkpoint_callback)

    return None


def compose_model_config(config_dir: str, model_name: str) -> DictConfig:
    """
    Re-compose a Hydra DictConfig for a specific model, keeping all CLI overrides.
    - Forces model=<model_name>
    - Keeps user's CLI overrides (except any existing model=...)
    - Keeps the current choices for dataset/experiment/lbw/pw/folds/path if the CLI did not set them
    """
    # 1) Grab current task overrides (the exact CLI the user passed)
    try:
        cli_overrides: List[str] = list(HydraConfig.get().overrides.task)
    except Exception:
        cli_overrides = []

    # 2) Remove any existing 'model=' override; we'll inject our own
    filtered = [ov for ov in cli_overrides if not ov.startswith("model=")]

    # 3) Ensure key group choices persist if the user didn’t set them explicitly
    try:
        choices = dict(HydraConfig.get().runtime.choices)
    except Exception:
        choices = {}

    for grp in ("dataset", "experiment", "lbw", "pw", "folds", "path"):
        if grp in choices and not any(ov.startswith(f"{grp}=") for ov in filtered):
            filtered.append(f"{grp}={choices[grp]}")

    cfg_name = HydraConfig.get().job.config_name or "config"

    return compose(config_name=cfg_name, overrides=[f"model={model_name}", *filtered])


def load_best_into(pl_model: LightningModule, ckpt_path: str) -> LightningModule:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    pl_model.load_state_dict(ckpt["state_dict"], strict=True)
    pl_model.eval()
    return pl_model


def train_test_global_ensemble(
    config: DictConfig, wandb_logger: WandbLogger, run_name: str
) -> None:
    fitted_models = []

    for model in config.ensemble_models:
        model_config = compose_model_config(config.path.config_path, model)

        datamodule, pl_model, trainer, callbacks = setup(
            model_config, DummyLogger(), run_name
        )
        print(f"Start Training Model {model}.")
        trainer.fit(pl_model, datamodule=datamodule)
        print(f"End Training Model {model}.")

        ckpt_cb = callbacks[0]
        if model not in config.special_models:
            best_path = ckpt_cb.best_model_path
            load_best_into(pl_model, best_path)
            delete_checkpoint(trainer, ckpt_cb)

        pl_model.eval()
        fitted_models.append(pl_model)

        del datamodule, trainer, callbacks
        torch.cuda.empty_cache()

    datamodule, pl_model, trainer, callbacks = setup(
        config, wandb_logger, run_name, fitted_models=fitted_models
    )
    trainer.fit(pl_model, datamodule=datamodule)
    trainer.test(pl_model, datamodule=datamodule, ckpt_path="best")


def train_test_local(
    config: DictConfig, wandb_logger: WandbLogger, run_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    test_participants_global = config.dataset.datamodule.test_participants
    global_specific_results = []
    results = []
    n_test_windows: List[int] = []
    for participant in config.dataset.participants:
        print(f"Participant {participant}.")
        datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
        datamodule.participant = participant
        checkpoint_callback = callbacks[0]

        print(f"Start Training {participant}.")
        trainer.fit(pl_model, datamodule=datamodule)
        print(f"End Training {participant}")

        print(f"Start Evaluation {participant}")
        if config.model.name not in config.special_models:
            test_trainer = trainer
            if test_trainer.is_global_zero:
                test_results = test_trainer.test(
                    pl_model, datamodule=datamodule, ckpt_path="best"
                )
        else:
            print("Best checkpoint not found, testing with current model.")
            test_trainer = trainer
            test_results = trainer.test(pl_model, datamodule=datamodule, ckpt_path=None)

        results.append(test_results[0])
        if participant in test_participants_global:
            global_specific_results.append(test_results[0])
            n_test_windows.append(len(datamodule.test_dataset))

        delete_checkpoint(test_trainer, checkpoint_callback)

        print(f"End Evaluation {participant}")

        del datamodule, pl_model, trainer, callbacks
        gc.collect()
        torch.cuda.empty_cache()

    # Local results
    df = pd.DataFrame(results)
    means = df.mean().to_frame().T
    stds = df.std().to_frame().T

    wandb_logger.experiment.log(
        {
            "mean_metrics": wandb.Table(dataframe=means),
            "std_metrics": wandb.Table(dataframe=stds),
            "raw_metrics": wandb.Table(dataframe=df),
        }
    )

    # Global results
    total_windows = sum(n_test_windows)
    global_averaged = defaultdict(list)
    for metric_dict, n_windows in zip(global_specific_results, n_test_windows):
        for metric_name, metric_value in metric_dict.items():
            global_averaged[metric_name].append(
                metric_value * (n_windows / total_windows)
            )

    global_averaged_means = {
        metric_name: np.sum(metric_values)
        for metric_name, metric_values in global_averaged.items()
    }

    means = pd.Series(global_averaged_means).to_frame().T

    wandb_logger.experiment.log(
        {
            "global_mean_metrics": wandb.Table(dataframe=means),
        }
    )

    return means, df
