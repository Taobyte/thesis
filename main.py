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
    wandb_logger = (
        WandbLogger(
            config=config_dict,
            project="thesis",
            log_model=True,
            save_code=True,
            reinit=True,
        )
        if config.use_wandb
        else DummyLogger()
    )

    if config.model.model_name == "elastst":
        data_manager = instantiate(config.model.data.data_manager)
        datamodule = instantiate(
            config.model.data.datamodule, data_manager=data_manager
        )
        model = instantiate(
            config.model.model,
            target_dim=data_manager.target_dim,
            context_length=data_manager.context_length,
            prediction_length=data_manager.prediction_length,
            freq=data_manager.freq,
            lags_list=data_manager.lags_list,
        )
    else:
        datamodule = instantiate(
            config.dataset.datamodule,
            batch_size=config.model.data.batch_size,
        )
        model = instantiate(config.model.model)

    pl_model = instantiate(config.model.pl_model, model=model)

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
