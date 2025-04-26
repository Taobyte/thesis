import yaml

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger


def setup_wandb_logger(config: DictConfig) -> WandbLogger:
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

    name = f"{config.dataset.name}_{config.model.name}_{signal_type}_{activity}_{config.look_back_window}_{config.prediction_window}"
    tags = [config.dataset.name, config.model.name, signal_type, activity]
    wandb_logger = (
        WandbLogger(
            name=name,
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

    return wandb_logger
