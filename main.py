import os
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
)
from src.models.utils import get_model_kwargs


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver(
    "suffix_if_true", lambda flag, suffix: suffix if flag else ""
)
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
        num_sanity_val_steps=0,
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
        best_ckpt_path = (
            checkpoint_callback.best_model_path
        )  # TODO not correct at the moment!
        if os.path.exists(best_ckpt_path):
            print(f"Best checkpoint found! best_ckpt_path: {best_ckpt_path}")
            trainer.test(datamodule=datamodule, ckpt_path="best")
        else:
            print("Best checkpoint not found, testing with current model.")
            trainer.test(pl_model, datamodule=datamodule, ckpt_path=None)
        print("End Evaluation.")


if __name__ == "__main__":
    main()
