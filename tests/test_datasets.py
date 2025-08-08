import torch

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm
from torch import Tensor
from torch.utils.data import DataLoader
from typing import Tuple

from src.utils import (
    compute_square_window,
    compute_input_channel_dims,
    get_optuna_name,
)

from src.normalization import global_z_denorm

OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
OmegaConf.register_new_resolver(
    "compute_input_channel_dims", compute_input_channel_dims
)


def _test_specific_dataset(
    dataset_name: str = "ieee",
    experiment: str = "endo_only",
    normalization: str = "global",
):
    look_back_windows = [5, 10, 20, 30, 60]
    prediction_windows = [3, 5, 10, 20, 30]

    for look_back_window, prediction_window in tqdm(
        zip(look_back_windows, prediction_windows),
        total=len(look_back_windows) * len(prediction_windows),
    ):
        print(f"[INFO] Test config → lbw={look_back_window}, pw={prediction_window}")

        overrides = [f"dataset={dataset_name}", f"experiment={experiment}"]
        overrides += [
            f"look_back_window={look_back_window}",
            f"prediction_window={prediction_window}",
            f"normalization={normalization}",
        ]

        with initialize(version_base=None, config_path="../config/"):
            cfg = compose(config_name="config", overrides=overrides)

        datamodule = instantiate(cfg.dataset.datamodule)
        datamodule.setup("fit")
        train_dl = datamodule.train_dataloader()
        val_dl = datamodule.val_dataloader()
        datamodule.setup("test")
        test_dl = datamodule.test_dataloader()

        look_back_channel_dim = datamodule.look_back_channel_dim
        target_channel_dim = datamodule.target_channel_dim

        def _test_dl(dl: DataLoader[Tuple[Tensor, Tensor, Tensor]]):
            for batch in dl:
                lbw, lbw_norm, pw_norm = batch
                t_x, t_y = lbw_norm.shape[1], pw_norm.shape[1]
                c_x, c_y = lbw_norm.shape[2], pw_norm.shape[2]

                # test shapes
                assert t_x == look_back_window and t_y == prediction_window, (
                    f"t_x: {t_x}, t_y: {t_y}"
                )
                assert c_x == look_back_channel_dim and c_y == target_channel_dim, (
                    f"c_x: {c_x}, c_y: {c_y}"
                )
                # test NaN or Inf
                assert torch.isfinite(lbw_norm).all(), (
                    "Non-finite (NaN or Inf) values found in lbw_norm"
                )
                assert torch.isfinite(pw_norm).all(), (
                    "Non-finite (NaN or Inf) values found in pw_norm"
                )
                assert torch.isfinite(lbw).all(), (
                    "Non-finite (NaN or Inf) values found in look_back_window"
                )

                # check type
                assert lbw.dtype == torch.float32
                assert lbw_norm.dtype == torch.float32
                assert pw_norm.dtype == torch.float32

                # check correct normalization
                if normalization == "global":
                    mean, std = (
                        datamodule.train_dataset.mean,
                        datamodule.train_dataset.std,
                    )
                    device = lbw.device
                    mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
                    std = torch.tensor(std).reshape(1, 1, -1).to(device).float()
                    denorm = global_z_denorm(
                        lbw_norm, datamodule.local_norm_channels, mean, std
                    )
                    assert torch.allclose(denorm, lbw, atol=1e-6), (
                        "Denormalized values not close to original"
                    )
                elif normalization == "none":
                    assert lbw == lbw_norm, (
                        "Lookback window is not equal to the normalized lookback window!"
                    )
                elif normalization == "difference":
                    assert lbw_norm[:, 0, :] == 0.0, (
                        "First normed lookback window value for differencing must be 0.0."
                    )

        _test_dl(train_dl)
        _test_dl(val_dl)
        _test_dl(test_dl)


def test_dalia():
    _test_specific_dataset("dalia", experiment="endo_only")
    _test_specific_dataset("dalia", experiment="endo_exo")


def test_ieee():
    _test_specific_dataset("ieee", experiment="endo_only")
    _test_specific_dataset("ieee", experiment="endo_exo")


def test_wildppg():
    _test_specific_dataset("wildppg", experiment="endo_only")
    _test_specific_dataset("wildppg", experiment="endo_exo")
