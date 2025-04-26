import yaml
import numpy as np

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from typing import Tuple


def setup_wandb_logger(config: DictConfig) -> Tuple[WandbLogger, str]:
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    hr_or_ppg = "hr" if config.use_heart_rate else "ppg"

    dataset_to_signal_type = {
        "dalia": hr_or_ppg,
        "wildppg": hr_or_ppg,
        "ieee": hr_or_ppg,
        "chapman": "ecg",
        "capture24": "acc",
        "ucihar": "acc",
        "usc": "acc",
    }

    signal_type = dataset_to_signal_type[config.dataset.name]
    activity = "activity" if config.use_activity_info else "no_activity"

    group_name = f"{config.dataset.name}_{signal_type}_{activity}_{config.look_back_window}_{config.prediction_window}"
    run_name = f"{config.dataset.name}_{config.model.name}_{signal_type}_{activity}_{config.look_back_window}_{config.prediction_window}"

    tags = [config.dataset.name, config.model.name, signal_type, activity]

    wandb_logger = (
        WandbLogger(
            name=run_name,
            group=group_name,
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

    return wandb_logger, run_name


# OmegaConf custom resolver functions


# this function is needed, because AdaMSHyper does not support short look_back_window lengths
# the custom resolver function is used in the model definition config/model/adamshyper.yaml
def compute_square_window(seq_len, max_window=4):
    """
    Finds the largest equal factors [k,k] such that k*k <= seq_len.
    Defaults to [4,4] if to valid window exists.
    """
    max_k = int(np.sqrt(seq_len))
    max_k = min(max_k, max_window)
    return [max_k, max_k]
