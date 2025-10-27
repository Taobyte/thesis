import numpy as np

from typing import Any, List
from numpy.typing import NDArray

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


class IEEEDataset(HRDataset):
    def __init__(
        self,
        sensor_location: str = "wrist",
        imu_features: list[str] = ["mean", "std"],
        **kwargs: Any,
    ):
        self.sensor_location = sensor_location
        self.imu_features = imu_features
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> List[NDArray[np.float32]]:
        loaded_series: list[NDArray[np.float32]] = []

        for participant in self.participants:
            data = np.load(self.data_dir + f"IEEE_{participant}.npz", allow_pickle=True)
            bpm = data["bpms"]
            series = bpm

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


class IEEEDataModule(BaseDataModule):
    def __init__(
        self,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7],
        val_participants: list[int] = [8, 9],
        test_participants: list[int] = [10, 11, 12],
        sensor_location: str = "wrist",
        imu_features: list[str] = ["mean"],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.imu_features = imu_features
        self.sensor_location = sensor_location

        self.common_args: dict[str, Any] = dict(
            data_dir=self.data_dir,
            look_back_window=self.look_back_window,
            prediction_window=self.prediction_window,
            use_dynamic_features=self.use_dynamic_features,
            target_channel_dim=self.target_channel_dim,
            test_local=self.test_local,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            imu_features=self.imu_features,
            sensor_location=self.sensor_location,
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
            self.val_dataset = IEEEDataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                participants=self.test_participants,
                return_whole_series=False,
                **self.common_args,
            )
