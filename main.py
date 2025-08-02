# The overall project structure and training loop design in this repository
# were significantly inspired by the work of Alice Bizeul.
# See their repository at: https://github.com/alicebizeul/pmae
import hydra
import numpy as np
import lightning as L

from omegaconf import DictConfig, OmegaConf
from hydra.utils import get_original_cwd
from typing import Optional

from src.utils import (
    setup,
    delete_checkpoint,
    setup_wandb_logger,
    get_optuna_name,
    compute_square_window,
    compute_input_channel_dims,
    get_min,
    resolve_str,
)


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)
OmegaConf.register_new_resolver("min", get_min)
OmegaConf.register_new_resolver("str", resolve_str)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    assert config.dataset.name in ["ieee", "dalia", "wildppg"]
    L.seed_everything(config.seed)
    wandb_logger, run_name = setup_wandb_logger(config)

    # print(OmegaConf.to_yaml(config))
    if config.tune:
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

            datamodule, pl_model, trainer, callbacks = setup(
                config, wandb_logger, run_name
            )
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

            if hasattr(pl_model, "cleanup_jax_memory"):
                pl_model.cleanup_jax_memory()

            delete_checkpoint(trainer, checkpoint_callback)

            del datamodule, pl_model, trainer, callbacks

        avg_val_loss = float(np.mean(val_losses))
        print(f"Average validation loss across folds: {avg_val_loss:.4f}")
        return avg_val_loss

    else:
        datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
        checkpoint_callback = callbacks[0]

        print("Start Training.")
        trainer.fit(pl_model, datamodule=datamodule)
        print("End Training.")

        print("Start Evaluation.")
        if config.model.name not in config.special_models:
            test_trainer = trainer
            if test_trainer.is_global_zero:
                test_trainer.test(pl_model, datamodule=datamodule, ckpt_path="best")

        else:
            print("Best checkpoint not found, testing with current model.")
            test_trainer = trainer
            trainer.test(pl_model, datamodule=datamodule, ckpt_path=None)

        delete_checkpoint(test_trainer, checkpoint_callback)

        print("End Evaluation.")


if __name__ == "__main__":
    main()
