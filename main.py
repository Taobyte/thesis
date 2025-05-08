import hydra
import lightning as L

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional

from src.utils import setup_wandb_logger, compute_square_window


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "suffix_if_true", lambda flag, suffix: suffix if flag else ""
)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    L.seed_everything(config.seed)
    wandb_logger, run_name = setup_wandb_logger(config)

    datamodule = instantiate(
        config.dataset.datamodule,
        batch_size=config.model.data.batch_size,
    )

    model_kwargs = {}
    if config.model.name == "gp":
        model_kwargs["inducing_points"] = datamodule.get_inducing_points(
            config.model.n_points
        )
        model_kwargs["train_dataset_length"] = datamodule.get_train_dataset_length()

    model = instantiate(config.model.model, **model_kwargs)
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

    if config.tune:
        val_results = trainer.validate(pl_model, datamodule=datamodule)

        last_val_loss = val_results[0]["val_loss_epoch"]
        # return val loss for tuner
        return last_val_loss
    else:
        print("Start Evaluation.")
        try:
            trainer.test(pl_model, datamodule=datamodule, ckpt_path="best")
        except FileNotFoundError:
            print("Best checkpoint not found, testing with current model.")
            trainer.test(pl_model, datamodule=datamodule, ckpt_path=None)
        print("End Evaluation.")


if __name__ == "__main__":
    main()
