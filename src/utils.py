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
    look_back_window: int,
    prediction_window: int,
    use_dynamic_features: bool,
    use_static_features: bool,
    fold_nr: int = 0,
    fold_datasets: list[str] = None,
    normalization: str = "local",
) -> Tuple[str, str, list[str]]:
    hr_or_ppg = "hr" if use_heart_rate else "ppg"
    dataset_to_signal_type = {
        "dalia": hr_or_ppg,
        "wildppg": hr_or_ppg,
        "ieee": hr_or_ppg,
        "mhc6mwt": "hr",
        "chapman": "ecg",
        "ptbxl": "ecg",
        "capture24": "acc",
        "ucihar": "acc",
        "usc": "acc",
    }

    signal_type = dataset_to_signal_type[dataset_name]

    # static and dynamic features
    features = ""
    if use_dynamic_features and use_static_features:
        features = "df_sf"
    elif use_dynamic_features and not use_static_features:
        features = "df"
    elif not use_dynamic_features and use_static_features:
        features = "sf"

    fold = ""
    if dataset_name in fold_datasets:
        fold = f"fold_{fold_nr}_"

    group_name = f"{normalization}_{dataset_name}_{signal_type}_{features}_{look_back_window}_{prediction_window}"
    run_name = f"{normalization}_{fold}{dataset_name}_{model_name}_{signal_type}_{features}_{look_back_window}_{prediction_window}"

    tags = [dataset_name, model_name, signal_type, normalization]
    if features != "":
        tags.append(features)
    if dataset_name in fold_datasets:
        tags.append(fold)

    return group_name, run_name, tags


def get_optuna_name(
    dataset_name: str,
    model_name: str,
    use_heart_rate: bool,
    look_back_window: int,
    prediction_window: int,
    use_dynamic_features: bool,
    use_static_features: bool,
    fold_nr: int = 0,
    fold_datasets: list[str] = None,
    normalization: str = "local",
):
    group_name, _, tags = create_group_run_name(
        dataset_name,
        model_name,
        use_heart_rate,
        look_back_window,
        prediction_window,
        use_dynamic_features,
        use_static_features,
        fold_nr,
        fold_datasets,
        normalization,
    )
    # tags[1] stores the models name
    return f"optuna_{tags[1]}_{group_name}"


def setup_wandb_logger(config: DictConfig) -> Tuple[WandbLogger, str]:
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    group_name, run_name, tags = create_group_run_name(
        config.dataset.name,
        config.model.name,
        config.use_heart_rate,
        config.look_back_window,
        config.prediction_window,
        config.use_dynamic_features,
        config.use_static_features,
        config.experiment.fold_nr,
        config.fold_datasets,
        config.normalization,
    )

    # print(config_dict)
    if config.tune:
        config.use_wandb = False
        return DummyLogger(), run_name

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
    print("WanDB Setup complete.")
    print(f"Run name: {run_name}")
    print(f"Group name: {group_name}")
    print(f"Tags: {tags}")

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


def compute_input_channel_dims(
    target_channel_dim: int,
    dynamic_exogenous_variables: int,
    static_exogenous_variables: int,
    use_dynamic_features: bool,
    use_static_features: bool,
) -> int:
    dims = target_channel_dim
    if use_dynamic_features:
        dims += dynamic_exogenous_variables
    if use_static_features:
        dims += static_exogenous_variables
    return dims
