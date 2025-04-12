import sys
from pathlib import Path

# Add the root directory to Python path
root_dir = str(Path(__file__).parent.parent.parent)
sys.path.insert(0, root_dir)

from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm


dataset_params = {
    "dalia": {
        "base_dim": 1,  # PPG or heart rate
        "heart_rate": True,
        "activity": True,
        "is_categorical": False,
        "n_activities": None,
    },
    "wildppg": {
        "base_dim": 1,  # PPG or heart rate
        "heart_rate": True,
        "activity": True,
        "is_categorical": False,
        "n_activities": None,
    },
    "ieee": {
        "base_dim": 1,
        "heart_rate": True,
        "activity": False,
        "is_categorical": None,
        "n_activities": None,
    },  # PPG or heart rate
    "chapman": {
        "base_dim": 4,  # ecg for 4 different locations on body
        "heart_rate": False,
        "activity": True,
        "is_categorical": True,
        "n_activities": 4,
    },
    "ucihar": {
        "base_dim": 561,  # gyro + acceleration + derived features
        "heart_rate": False,
        "activity": True,
        "is_categorical": True,
        "n_activities": 6,
    },
    "usc": {
        "base_dim": 6,  # 3 gyro + 3 acceleration
        "heart_rate": False,
        "activity": True,
        "is_categorical": True,
        "n_activities": 12,
    },
    "capture24": {
        "base_dim": 3,  # acceleration in x,y,z direction
        "heart_rate": False,
        "activity": True,
        "is_categorical": False,
        "n_activities": None,
    },
}


def _test_specific_dataset(
    name: str,
    base_dim: int,
    heart_rate: bool,
    activity: bool,
    is_categorical: bool,
    n_activities: int,
):
    look_back_windows = [16, 64, 128]
    prediction_windows = [1, 8, 32]

    hr_flags = [False, True] if heart_rate else [False]
    activity_flags = [False, True] if activity else [False]

    for look_back_window, prediction_window in tqdm(
        zip(look_back_windows, prediction_windows),
        total=len(look_back_windows) * len(prediction_windows),
    ):
        for hr_flag in tqdm(hr_flags, total=len(hr_flags)):
            for activity_flag in tqdm(activity_flags, total=len(activity_flags)):
                overrides = [f"dataset={name}"]
                overrides += [
                    f"look_back_window={look_back_window}",
                    f"prediction_window={prediction_window}",
                ]

                if hr_flag:
                    overrides += ["dataset.datamodule.use_heart_rate=True"]
                activity_channels = 0
                if activity_flag:
                    overrides += ["dataset.datamodule.use_activity_info=True"]
                    activity_channels = n_activities if is_categorical else 1

                with initialize(version_base=None, config_path="../../config/"):
                    cfg = compose(config_name="config", overrides=overrides)

                datamodule = instantiate(cfg.dataset)["datamodule"]
                datamodule.setup("fit")
                train_dl = datamodule.train_dataloader()
                val_dl = datamodule.val_dataloader()
                datamodule.setup("test")
                test_dl = datamodule.test_dataloader()

                def _test_dl(dl):
                    for batch in dl:
                        x, y = batch
                        t_x, t_y = x.shape[1], y.shape[1]
                        c_x, c_y = x.shape[2], y.shape[2]

                        assert t_x == look_back_window and t_y == prediction_window
                        assert c_x == base_dim + activity_channels and c_y == base_dim
                        break

                _test_dl(train_dl)
                _test_dl(val_dl)
                _test_dl(test_dl)


def test_dalia():
    _test_specific_dataset("dalia", **dataset_params["dalia"])


def test_ieee():
    _test_specific_dataset("ieee", **dataset_params["ieee"])
