import hydra
import yaml
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger


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
    datamodule.setup("fit")  # already setup for mean and std info

    global_mean = datamodule.train_dataset.global_mean
    global_std = datamodule.train_dataset.global_std

    model = instantiate(config.model.model)
    pl_model = instantiate(
        config.model.pl_model,
        model=model,
        global_mean=global_mean,
        global_std=global_std,
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

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.model.trainer.max_epochs,
        enable_progress_bar=True,
        enable_model_summary=True,
        overfit_batches=1 if config.overfit else 0.0,
    )

    print("Start Training.")
    trainer.fit(pl_model, datamodule=datamodule)
    print("End Training.")

    print("Start Evaluation.")
    trainer.test(datamodule=datamodule)
    print("End Evaluation.")


if __name__ == "__main__":
    main()
