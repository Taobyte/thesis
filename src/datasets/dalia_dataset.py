import numpy as np

from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from typing import Any

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


class DaLiADataset(HRDataset):
    def __init__(self, window_statistic: str = "var", **kwargs: Any):
        self.window_statistic = window_statistic
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        loaded_series: list[NDArray[np.float32]] = []
        for i in self.participants:
            label = "S" + str(i)
            data_path = Path(self.data_dir) / (label + ".npz")
            data = np.load(data_path)
            series = (
                data["heart_rate"][:, np.newaxis]
                if self.use_heart_rate
                else data["bvp"]
            )

            if self.use_dynamic_features:
                activity = data["acc_norm_ppg"][:, np.newaxis]
                if self.use_heart_rate:
                    if self.window_statistic == "mean":
                        activity = data["acc_norm_heart_rate"][:, np.newaxis]
                    elif self.window_statistic == "var":
                        activity = data["imu_var"][:, np.newaxis]
                    elif self.window_statistic == "power":
                        activity = data["imu_power"][:, np.newaxis]
                    else:
                        raise NotImplementedError()

                series = np.concatenate((series, activity), axis=1)

            if self.use_static_features:
                activity = data["activity"]
                if self.use_heart_rate:
                    encoder = OneHotEncoder(
                        categories=[list(range(0, 9))], sparse_output=False
                    )
                    window_size = 8
                    stride = 2
                    argmax_activities: list[int] = []
                    activity_label_1hz = activity[::4].astype(
                        np.int64
                    )  # activity is 4Hz
                    for i in range(
                        0, len(activity_label_1hz) - window_size + 1, stride
                    ):
                        window = activity_label_1hz[i : i + window_size, :]
                        counts = np.bincount(window[:, 0], minlength=9)
                        argmax_activities.append(int(np.argmax(counts)))

                    processed_activity = np.array(argmax_activities)[:, np.newaxis]
                    assert len(processed_activity) == len(series)

                    processed_activity_onehot = encoder.fit_transform(
                        processed_activity
                    )
                else:
                    processed_activity = 0  # TODO
                series = np.concatenate((series, processed_activity_onehot), axis=1)

            loaded_series.append(series)

        combined = np.concatenate(loaded_series, axis=0)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        min = np.min(combined, axis=0)
        max = np.min(combined, axis=0)

        return loaded_series, [mean, std, min, max]


class DaLiADataModule(BaseDataModule):
    def __init__(
        self,
        use_heart_rate: bool = False,
        train_participants: list[int] = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list[int] = [9, 10, 11],
        test_participants: list[int] = [1, 13, 14],
        window_statistic: str = "mean",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate
        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.window_statistic = window_statistic

    def setup(self, stage: str = "fit"):
        common_args = {
            "data_dir": self.data_dir,
            "use_dynamic_features": self.use_dynamic_features,
            "use_static_features": self.use_static_features,
            "use_heart_rate": self.use_heart_rate,
            "look_back_window": self.look_back_window,
            "prediction_window": self.prediction_window,
            "target_channel_dim": self.target_channel_dim,
            "window_statistic": self.window_statistic,
            "test_local": self.test_local,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
        }

        if stage == "fit":
            self.train_dataset = DaLiADataset(
                participants=self.train_participants,
                **common_args,
            )
            self.val_dataset = DaLiADataset(
                participants=self.val_participants,
                **common_args,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                participants=self.test_participants,
                **common_args,
            )
