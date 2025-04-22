from hydra import initialize, compose
from hydra.utils import instantiate
from omegaconf import OmegaConf
from tqdm import tqdm

OmegaConf.register_new_resolver("eval", eval)


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
        "activity": True,
        "is_categorical": None,
        "n_activities": None,
    },  # PPG or heart rate
    "chapman": {
        "base_dim": 4,  # ecg for 4 different locations on body
        "heart_rate": False,
        "activity": True,
        "is_categorical": True,
        "n_activities": 4,
        "activity_flag_name": "use_disease",
    },
    "ucihar": {
        "base_dim": 9,  # gyro + acceleration + derived features
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
    activity_flag_name: str = None,
):
    look_back_windows = [16, 64, 100]
    prediction_windows = [1, 8, 28]

    hr_flags = [False, True] if heart_rate else [False]
    activity_flags = [False, True] if activity else [False]

    for look_back_window, prediction_window in tqdm(
        zip(look_back_windows, prediction_windows),
        total=len(look_back_windows) * len(prediction_windows),
    ):
        for hr_flag in tqdm(hr_flags, total=len(hr_flags)):
            for activity_flag in tqdm(activity_flags, total=len(activity_flags)):
                print(
                    f"[INFO] Test config → lbw={look_back_window}, pw={prediction_window}, hr={hr_flag}, act={activity_flag}"
                )

                overrides = [f"dataset={name}"]
                overrides += [
                    f"look_back_window={look_back_window}",
                    f"prediction_window={prediction_window}",
                ]

                if hr_flag:
                    overrides += ["dataset.datamodule.use_heart_rate=True"]
                activity_channels = 0
                if activity_flag:
                    if activity_flag_name:
                        overrides += [f"dataset.datamodule.{activity_flag_name}=True"]
                    else:
                        overrides += ["dataset.datamodule.use_activity_info=True"]
                    activity_channels = n_activities if is_categorical else 1

                with initialize(version_base=None, config_path="../config/"):
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

                        assert t_x == look_back_window and t_y == prediction_window, (
                            f"t_x: {t_x}, t_y: {t_y}"
                        )
                        assert (
                            c_x == base_dim + activity_channels and c_y == base_dim
                        ), f"c_x: {c_x}, c_y: {c_y}"
                        break

                _test_dl(train_dl)
                _test_dl(val_dl)
                _test_dl(test_dl)


def test_dalia():
    _test_specific_dataset("dalia", **dataset_params["dalia"])


def test_ieee():
    _test_specific_dataset("ieee", **dataset_params["ieee"])


def test_chapman():
    _test_specific_dataset("chapman", **dataset_params["chapman"])


def test_usc():
    _test_specific_dataset("usc", **dataset_params["usc"])


def test_wildppg():
    _test_specific_dataset("wildppg", **dataset_params["wildppg"])


def test_ucihar():
    _test_specific_dataset("ucihar", **dataset_params["ucihar"])
