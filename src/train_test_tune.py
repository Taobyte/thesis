import numpy as np
import pandas as pd
import wandb

from lightning.pytorch.loggers import WandbLogger
from omegaconf import OmegaConf
from omegaconf import DictConfig
from hydra.utils import get_original_cwd
from typing import Tuple, List
from collections import defaultdict

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

    averaged_results = float(np.mean(results))
    return averaged_results


def train_test_global(
    config: DictConfig, wandb_logger: WandbLogger, run_name: str
) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
    local_datamodule, _, _, _ = setup(config, wandb_logger, run_name)
    local_datamodule.test_local = True
    local_datamodule.setup("fit")  # initialize train_dataset for normalization

    if config.model.name not in config.special_models:
        local_test_results = trainer.test(
            pl_model, datamodule=local_datamodule, ckpt_path="best"
        )
    else:
        print("Best checkpoint not found, testing with current model.")
        local_test_results = trainer.test(
            pl_model, datamodule=local_datamodule, ckpt_path=None
        )

    delete_checkpoint(trainer, checkpoint_callback)

    print("End Evaluation.")

    global_results = pd.DataFrame(global_test_results)
    local_results = pd.DataFrame(local_test_results)
    return global_results, local_results


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

        print("Start Evaluation.")
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

        print("End Evaluation.")

    # Local results
    df = pd.DataFrame(results)
    means = df.mean()
    stds = df.std()

    wandb_logger.experiment.log(
        {
            "mean_metrics": wandb.Table(dataframe=means.to_frame(name="mean")),
            "std_metrics": wandb.Table(dataframe=stds.to_frame(name="std")),
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

    means = pd.Series(global_averaged_means)

    wandb_logger.experiment.log(
        {
            "global_mean_metrics": wandb.Table(dataframe=means.to_frame(name="mean")),
            "global_std_metrics": wandb.Table(dataframe=stds.to_frame(name="std")),
        }
    )

    global_df = pd.DataFrame([means])

    return global_df, df
