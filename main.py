import hydra
import lightning as L

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint

from src.utils import setup_wandb_logger, compute_square_window


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig):
    L.seed_everything(config.seed)

    wandb_logger, run_name = setup_wandb_logger(config)

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
    if config.use_checkpoint_callback:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_loss",
            filename=run_name + "-{epoch}-{step}",
            save_top_k=1,  # Also saves best model if you want
        )

        callbacks.append(checkpoint_callback)

    # multi gpu training
    multi_gpu_dict = {}
    if config.use_multi_gpu:
        multi_gpu_dict = config.multi

    trainer = L.Trainer(
        logger=wandb_logger,
        max_epochs=config.model.trainer.max_epochs,
        callbacks=callbacks,
        enable_progress_bar=True,
        enable_model_summary=False,
        overfit_batches=1 if config.overfit else 0.0,
        limit_test_batches=10 if config.overfit else None,
        default_root_dir=config.path.basedir,
        **multi_gpu_dict,
    )

    print("Start Training.")
    trainer.fit(pl_model, datamodule=datamodule)
    print("End Training.")

    print("Start Evaluation.")
    trainer.test(datamodule=datamodule)
    print("End Evaluation.")


if __name__ == "__main__":
    main()
