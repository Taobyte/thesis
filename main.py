# The overall project structure and training loop design in this repository
# were significantly inspired by the work of Alice Bizeul.
# See their repository at: https://github.com/alicebizeul/pmae
import hydra
import lightning as L

from omegaconf import DictConfig, OmegaConf
from typing import Optional

from src.train_test_tune import (
    tune,
    tune_local,
    train_test_global,
    train_test_local,
)
from src.utils import (
    setup_wandb_logger,
    get_optuna_name,
    compute_square_window,
    compute_input_channel_dims,
    get_min,
    resolve_str,
    number_of_exo_vars,
)


OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)
OmegaConf.register_new_resolver("min", get_min)
OmegaConf.register_new_resolver("str", resolve_str)
OmegaConf.register_new_resolver("number_of_exo_vars", number_of_exo_vars)


@hydra.main(version_base="1.2", config_path="config", config_name="config.yaml")
def main(config: DictConfig) -> Optional[float]:
    L.seed_everything(config.seed)
    wandb_logger, run_name = setup_wandb_logger(config)

    # print(OmegaConf.to_yaml(config))
    if config.tune:
        if config.dataset.name in config.global_datasets:
            avg_val_loss = tune(config, wandb_logger, run_name)
        elif config.dataset.name in config.local_datasets:
            avg_val_loss = tune_local(config, wandb_logger, run_name)
        else:
            raise NotImplementedError()
        return avg_val_loss
    else:
        if config.dataset.name in config.global_datasets:
            train_test_global(config, wandb_logger, run_name)
        elif config.dataset.name in config.local_datasets:
            train_test_local(config, wandb_logger, run_name)
        else:
            raise NotImplementedError()


if __name__ == "__main__":
    main()
