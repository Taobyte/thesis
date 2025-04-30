import yaml
import numpy as np

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from typing import Tuple


def create_group_run_name(
    dataset_name: str,
    model_name: str,
    use_heart_rate: bool,
    use_activity_info: bool,
    look_back_window: int,
    prediction_window: int,
) -> Tuple[str, str, list[str]]:
    hr_or_ppg = "hr" if use_heart_rate else "ppg"

    dataset_to_signal_type = {
        "dalia": hr_or_ppg,
        "wildppg": hr_or_ppg,
        "ieee": hr_or_ppg,
        "chapman": "ecg",
        "capture24": "acc",
        "ucihar": "acc",
        "usc": "acc",
    }

    signal_type = dataset_to_signal_type[dataset_name]
    activity = "activity" if use_activity_info else "no_activity"

    group_name = f"{dataset_name}_{signal_type}_{activity}_{look_back_window}_{prediction_window}"
    run_name = f"{dataset_name}_{model_name}_{signal_type}_{activity}_{look_back_window}_{prediction_window}"

    tags = [dataset_name, model_name, signal_type, activity]

    return group_name, run_name, tags


def setup_wandb_logger(config: DictConfig) -> Tuple[WandbLogger, str]:
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    # print(config_dict)
    if config.tune:
        config.use_wandb = False

    group_name, run_name, tags = create_group_run_name(
        config.dataset.name,
        config.model.name,
        config.use_heart_rate,
        config.use_activity_info,
        config.look_back_window,
        config.prediction_window,
    )

    wandb_logger = (
        WandbLogger(
            name=run_name,
            group=group_name,
            config=config_dict,
            tags=tags,
            **config.wandb,
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
