import os
import numpy as np
import hydra
import yaml
import lightning as L

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from lightning.pytorch.callbacks import ModelCheckpoint


def compute_square_window(seq_len, max_window=4):
    """
    Finds the largest equal factors [k,k] such that k*k <= seq_len.
    Defaults to [4,4] if to valid window exists.
    """
    max_k = int(np.sqrt(seq_len))  # Largest possible equal factor
    max_k = min(max_k, max_window)  # Respect user's max_window
    return [max_k, max_k]


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    # print(config)
    L.seed_everything(config.seed)
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    signal_type = "hr" if config.dataset.datamodule.use_heart_rate else "ppg"
    activity = (
        "activity" if config.dataset.datamodule.use_activity_info else "no_activity"
    )

    name = f"{config.dataset.name}_{config.model.name}_{signal_type}_{activity}_{config.look_back_window}_{config.prediction_window}"
    tags = [config.dataset.name, config.model.name, signal_type, activity]
    wandb_logger = (
        WandbLogger(
            name=name,
            config=config_dict,
            project="thesis",
            log_model=True,
            save_code=True,
            reinit=True,
            tags=tags,
        )
        if config.use_wandb
        else DummyLogger()
    )

    datamodule = instantiate(
        config.dataset.datamodule,
        batch_size=config.model.data.batch_size,
    )

    model = instantiate(config.model.model)
    pl_model = instantiate(
        config.model.pl_model,
        model=model,
    )

    callbacks = []
    if config.model.trainer.use_early_stopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=config.model.trainer.patience,
            )
        )

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        filename="last-{epoch}",
        save_last=True,  # Saves a 'last.ckpt' file
        save_top_k=1,  # Also saves best model if you want
    )

    callbacks.append(checkpoint_callback)

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.model.trainer.max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=True,
        overfit_batches=1 if config.overfit else 0.0,
        limit_test_batches=10 if config.overfit else None,
    )

    print("Start Training.")
    trainer.fit(pl_model, datamodule=datamodule)
    print("End Training.")

    print("Start Evaluation.")
    trainer.test(ckpt_path="last", datamodule=datamodule)
    print("End Evaluation.")


if __name__ == "__main__":
    main()
