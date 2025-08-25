import numpy as np

from hydra import initialize, compose
from omegaconf import OmegaConf
from lightning.pytorch.loggers.logger import DummyLogger

from src.utils import (
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
)

from src.train_test_tune import train_test_global, train_test_local

OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)


def _test_specific_dataset(
    dataset_name: str = "ieee",
    experiment: str = "endo_only",
    normalization: str = "none",
):
    wandb_logger = DummyLogger()
    overrides = [f"dataset={dataset_name}", f"experiment={experiment}", "model=dummy"]
    overrides += [
        "look_back_window=5",
        "prediction_window=3",
        f"normalization={normalization}",
    ]

    with initialize(version_base=None, config_path="../config/"):
        cfg = compose(config_name="config", overrides=overrides)

    _, g_res_l = train_test_global(cfg, wandb_logger, "")

    overrides = [f"dataset=l{dataset_name}", f"experiment={experiment}", "model=dummy"]
    overrides += [
        "look_back_window=5",
        "prediction_window=3",
        f"normalization={normalization}",
    ]

    with initialize(version_base=None, config_path="../config/"):
        cfg = compose(config_name="config", overrides=overrides)

    l_res_g, _ = train_test_local(cfg, wandb_logger, "")

    g_values = g_res_l[["MSE", "MAE", "abs_target_mean", "naive_mae"]].values
    l_values = l_res_g[["MSE", "MAE", "abs_target_mean", "naive_mae"]].values

    import pdb

    pdb.set_trace()

    assert np.linalg.norm(g_values - l_values) <= 0.01


def test_dalia():
    _test_specific_dataset("dalia", experiment="endo_only")
    _test_specific_dataset("dalia", experiment="endo_exo")


def test_ieee():
    _test_specific_dataset("ieee", experiment="endo_only")
    _test_specific_dataset("ieee", experiment="endo_exo")


def test_wildppg():
    _test_specific_dataset("wildppg", experiment="endo_only")
    _test_specific_dataset("wildppg", experiment="endo_exo")
