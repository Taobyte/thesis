import hydra
import lightning as L

from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from typing import Optional

from src.utils import (
    setup_wandb_logger,
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
)
from src.models.utils import get_model_kwargs


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    L.seed_everything(config.seed)
    wandb_logger, run_name = setup_wandb_logger(config)

    datamodule = instantiate(config.dataset.datamodule)

    model_kwargs = get_model_kwargs(config, datamodule)
    model = instantiate(config.model.model, **model_kwargs)
    pl_model = instantiate(
        config.model.pl_model,
        model=model,
        name=config.model.name,
        use_plots=config.use_plots,
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
        # limit_test_batches=10 if config.overfit else None,
        default_root_dir=config.path.basedir,
        num_sanity_val_steps=0,
        **multi_gpu_dict,
    )

    if config.tune:
        import numpy as np
        from hydra.core.global_hydra import GlobalHydra
        from hydra import initialize_config_module, compose

        # loop over all folds and return average val loss performance
        val_losses = []
        for i in range(3):
            if config.dataset.name in config.fold_datasets:
                fold_name = f"fold_{i}"
                overrides = [f"experiment={fold_name}"]
            else:
                overrides = [f"seed={i}"]

            if GlobalHydra.instance().is_initialized():
                GlobalHydra.instance().clear()
            with initialize_config_module(version_base="1.1", config_module="config"):
                fold_config = compose(config_name="config.yaml", overrides=overrides)

            datamodule = instantiate(fold_config.dataset.datamodule)
            model_kwargs = get_model_kwargs(config, datamodule)
            model = instantiate(config.model.model, **model_kwargs)
            pl_model = instantiate(config.model.pl_model, model=model, tune=True)

            trainer = L.Trainer(
                logger=wandb_logger,
                max_epochs=config.model.trainer.max_epochs,
                callbacks=callbacks,
                enable_progress_bar=True,
                enable_model_summary=False,
                overfit_batches=1 if config.overfit else 0.0,
                limit_test_batches=10 if config.overfit else None,
                default_root_dir=config.path.basedir,
                num_sanity_val_steps=0,
                **multi_gpu_dict,
            )

            print(f"Starting fold {i}")
            trainer.fit(pl_model, datamodule=datamodule)
            if config.model.name not in ["xgboost", "bnn"]:
                val_results = trainer.validate(
                    datamodule=datamodule, ckpt_path="best"
                )  # use best model to validate
            else:
                val_results = trainer.validate(pl_model, datamodule=datamodule)
            last_val_loss = val_results[0]["val_loss_epoch"]
            val_losses.append(last_val_loss)
            print(f"Finished fold {i}, val_loss = {last_val_loss:.4f}")

        avg_val_loss = float(np.mean(val_losses))
        print(f"Average validation loss across folds: {avg_val_loss:.4f}")
        return avg_val_loss

    else:
        print("Start Training.")
        trainer.fit(pl_model, datamodule=datamodule)
        print("End Training.")

        print("Start Evaluation.")
        if config.model.name not in ["xgboost", "bnn"]:
            trainer.test(datamodule=datamodule, ckpt_path="best")
        else:
            print("Best checkpoint not found, testing with current model.")
            trainer.test(pl_model, datamodule=datamodule, ckpt_path=None)
        print("End Evaluation.")


if __name__ == "__main__":
    main()
