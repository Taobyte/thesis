import numpy as np

from numpy.typing import NDArray
from pathlib import Path
from typing import Any

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


class DaLiADataset(HRDataset):
    def __init__(
        self,
        sensor_location: str = "chest",
        imu_features: list[str] = ["mean", "std"],
        **kwargs: Any,
    ):
        self.sensor_location = sensor_location
        self.imu_features = imu_features
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> list[NDArray[np.float32]]:
        loaded_series: list[NDArray[np.float32]] = []
        for i in self.participants:
            label = "S" + str(i)
            data_path = Path(self.data_dir) / (label + ".npz")
            data = np.load(data_path)
            series = data["hr"][:, np.newaxis]

            if self.use_dynamic_features:
                features: list[NDArray[np.float32]] = []
                for feature in self.imu_features:
                    features.append(data[self.sensor_location + "_" + feature])
                assert len(features) > 0, (
                    "ATTENTION: you set use_dynamic_features=True, but pass no imu_features"
                )
                series = np.concatenate([series] + features, axis=1)

            loaded_series.append(series)

        return loaded_series

    def _read_activity_data_(self) -> list[NDArray[np.int32]]:
        participant_activity_labels: list[NDArray[np.int32]] = []
        for i in self.participants:
            label = "S" + str(i)
            data_path = Path(self.data_dir) / (label + ".npz")
            data = np.load(data_path)
            activity = data["activity"]
            activity_label_1hz = activity[::4].astype(np.int32)  # activity is 4Hz
            argmax_activities: list[int] = []
            window_size = 8
            stride = 2
            for i in range(0, len(activity_label_1hz) - window_size + 1, stride):
                window = activity_label_1hz[i : i + window_size, :]
                counts = np.bincount(window[:, 0], minlength=9)
                argmax_activities.append(int(np.argmax(counts)))

            processed_activity = np.array(argmax_activities)[:, np.newaxis]
            participant_activity_labels.append(processed_activity)

        return participant_activity_labels


class DaLiADataModule(BaseDataModule):
    def __init__(
        self,
        use_heart_rate: bool = False,
        train_participants: list[int] = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list[int] = [9, 10, 11],
        test_participants: list[int] = [1, 13, 14],
        sensor_location: str = "chest",
        imu_features: list[str] = ["mean", "std"],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate
        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.sensor_location = sensor_location
        self.imu_features = imu_features

        self.common_args: dict[str, Any] = {
            "data_dir": self.data_dir,
            "use_dynamic_features": self.use_dynamic_features,
            "use_static_features": self.use_static_features,
            "look_back_window": self.look_back_window,
            "prediction_window": self.prediction_window,
            "target_channel_dim": self.target_channel_dim,
            "test_local": self.test_local,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "sensor_location": self.sensor_location,
            "imu_features": self.imu_features,
        }

    def setup(self, stage: str = "fit"):
        if stage == "fit":
            self.train_dataset = DaLiADataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
            self.val_dataset = DaLiADataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                participants=self.test_participants,
                return_whole_series=False,
                **self.common_args,
            )
