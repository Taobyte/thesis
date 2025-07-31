from hydra import initialize, compose
from omegaconf import OmegaConf, DictConfig
from tqdm import tqdm
from pathlib import Path

from src.utils import (
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
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


def get_leaf_paths(config, parent_key=""):
    paths = []
    for key, value in config.items():
        new_key = f"{parent_key}.{key}" if parent_key else key

        # Handle OmegaConf DictConfig and regular dict
        if isinstance(value, (dict, DictConfig)):
            paths.extend(get_leaf_paths(value, new_key))
        else:
            paths.append(new_key)
    return paths


def compare_config_paths(ablation_cfg, cfg, verbose=True):
    ablation_paths = get_leaf_paths(ablation_cfg)

    for path in ablation_paths:
        cfg_value = OmegaConf.select(cfg, path)
        ablation_value = OmegaConf.select(ablation_cfg, path)

        if cfg_value is None:
            print(f"cfg_value does not exist for path {path}")
            return False
        elif cfg_value != ablation_value:
            print(f"cfg_value is not the same {cfg_value} != {ablation_value}")
            return False

    return True


def test_ablation():
    models = [
        "linear",
        "kalmanfilter",
        "dynamax",
        "gp",
        "xgboost",
        "timesnet",
        "simpletm",
        "adamshyper",
        "timexer",
        "gpt4ts",
    ]
    lbws = [str(i) for i in [5, 10, 20, 30, 60]]
    datasets = ["dalia", "ieee", "wildppg"]

    path = Path("C:/Users/cleme/ETH/Master/Thesis/ns-forecast/config/params")
    for model in models:
        for dataset in datasets:
            for lbw in lbws:
                config_path = path / model / dataset / f"{lbw}.yaml"

                if config_path.exists():
                    print(f"Processing {model} {dataset} {lbw}")
                    ablation_cfg = OmegaConf.load(config_path)
                    overrides = [
                        f"model={model}",
                        f"dataset={dataset}",
                        # f"look_back_window={lbw}",
                        f'lbw="{str(lbw)}"',
                    ]
                    with initialize(version_base=None, config_path="../config/"):
                        cfg = compose(config_name="config", overrides=overrides)

                    assert compare_config_paths(ablation_cfg, cfg), (
                        f"FAILED for {model} {dataset} {lbw}"
                    )
