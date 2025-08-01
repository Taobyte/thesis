import yaml
import numpy as np

from omegaconf import DictConfig, OmegaConf
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.loggers.logger import DummyLogger
from typing import Tuple, Union


def create_group_run_name(
    dataset_name: str,
    model_name: str,
    use_heart_rate: bool,
    look_back_window: int,
    prediction_window: int,
    fold_nr: int = 0,
    fold_datasets: list[str] = ["dalia", "wildppg", "ieee"],
    normalization: str = "global",
    experiment_name: str = "endo_only",
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

    fold = ""
    if dataset_name in fold_datasets:
        fold = f"fold_{fold_nr}_"

    group_name = f"{normalization}_{dataset_name}_{signal_type}_{experiment_name}_{look_back_window}_{prediction_window}"
    run_name = f"{normalization}_{fold}{dataset_name}_{model_name}_{signal_type}_{experiment_name}_{look_back_window}_{prediction_window}"

    tags = [dataset_name, model_name, signal_type, normalization, experiment_name]
    if dataset_name in fold_datasets:
        tags.append(fold)

    return group_name, run_name, tags


def get_optuna_name(
    dataset_name: str,
    model_name: str,
    use_heart_rate: bool,
    look_back_window: int,
    prediction_window: int,
    fold_nr: int = 0,
    fold_datasets: list[str] = ["dalia", "wildppg", "ieee"],
    normalization: str = "global",
    experiment_name: str = "endo_exo",
):
    group_name, _, tags = create_group_run_name(
        dataset_name,
        model_name,
        use_heart_rate,
        look_back_window,
        prediction_window,
        fold_nr,
        fold_datasets,
        normalization,
        experiment_name,
    )
    # tags[1] stores the models name
    return f"optuna_{tags[1]}_{group_name}"


def setup_wandb_logger(
    config: DictConfig,
) -> Tuple[Union[WandbLogger, DummyLogger], str]:
    config_dict = yaml.safe_load(OmegaConf.to_yaml(config, resolve=True))

    group_name, run_name, tags = create_group_run_name(
        config.dataset.name,
        config.model.name,
        config.use_heart_rate,
        config.look_back_window,
        config.prediction_window,
        config.folds.fold_nr,
        config.fold_datasets,
        config.normalization,
        config.experiment.experiment_name,
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
    print("Setup complete.")
    print(f"Run name: {run_name}")
    print(f"Group name: {group_name}")
    print(f"Tags: {tags}")

    return wandb_logger, run_name


# -------------------------------------------------------------------------------------------------------
# OmegaConf custom resolver functions
# -------------------------------------------------------------------------------------------------------


def get_min(x: str, y: str) -> int:
    return min(int(x), int(y))


def resolve_str(x: int) -> str:
    return str(x)


# this function is needed, because AdaMSHyper does not support short look_back_window lengths
# the custom resolver function is used in the model definition config/model/adamshyper.yaml
def compute_square_window(seq_len: int, max_window: int = 4) -> list[int]:
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
    use_only_exogenous_features: bool,
    use_perfect_info: bool,
) -> int:
    dims = target_channel_dim

    if use_only_exogenous_features:
        assert use_dynamic_features or use_static_features, (
            "Attention you are using only exogenous variables for training, but don't include exogenous variables"
        )
        dims = 0
    elif use_perfect_info:
        dims += 1
    else:
        dims = dims

    if use_dynamic_features:
        dims += dynamic_exogenous_variables
    if use_static_features:
        dims += static_exogenous_variables

    return dims
