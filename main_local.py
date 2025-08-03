# The overall project structure and training loop design in this repository
# were significantly inspired by the work of Alice Bizeul.
# See their repository at: https://github.com/alicebizeul/pmae
import hydra
import wandb
import lightning as L
import pandas as pd

from omegaconf import DictConfig, OmegaConf
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
    assert config.dataset.name in ["fdalia", "fwildppg", "fieee"]
    L.seed_everything(config.seed)
    wandb_logger, run_name = setup_wandb_logger(config)
    test_participants_global = config.dataset.datamodule.test_participants
    global_specific_results = []
    results = []
    for participant in config.dataset.participants:
        print(f"Participant {participant}.")
        datamodule, pl_model, trainer, callbacks = setup(config, wandb_logger, run_name)
        datamodule.participant = participant
        checkpoint_callback = callbacks[0]

        print("Start Training.")
        trainer.fit(pl_model, datamodule=datamodule)
        print("End Training.")

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

        delete_checkpoint(test_trainer, checkpoint_callback)

        print("End Evaluation.")

    df = pd.DataFrame(results)
    means = df.mean()
    stds = df.std()
    print(pd.DataFrame({"means": means, "stds": stds}))

    wandb_logger.experiment.log(
        {
            "mean_metrics": wandb.Table(dataframe=means.to_frame(name="mean")),
            "std_metrics": wandb.Table(dataframe=stds.to_frame(name="std")),
        }
    )

    global_df = pd.DataFrame(global_specific_results)
    global_means = global_df.mean()
    global_stds = global_df.std()

    wandb_logger.experiment.log(
        {
            "global_mean_metrics": wandb.Table(
                dataframe=global_means.to_frame(name="mean")
            ),
            "global_std_metrics": wandb.Table(
                dataframe=global_stds.to_frame(name="std")
            ),
        }
    )


if __name__ == "__main__":
    main()
