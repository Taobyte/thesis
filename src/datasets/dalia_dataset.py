import numpy as np
import torch
# import pandas as pd


from torch import Tensor
from torch.utils.data import Dataset
from numpy.typing import NDArray
from pathlib import Path
from typing import Tuple
from sklearn.preprocessing import OneHotEncoder
from typing import Any

from src.datasets.utils import BaseDataModule


def dalia_load_data(
    path: str,
    participants: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
    use_static_features: bool,
    window_statistic: str = "mean",
) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
    loaded_series: list[NDArray[np.float32]] = []
    for i in participants:
        label = "S" + str(i)
        data_path = Path(path) / (label + ".npz")
        data = np.load(data_path)
        series = data["heart_rate"][:, np.newaxis] if use_heart_rate else data["bvp"]

        if use_dynamic_features:
            activity = data["acc_norm_ppg"][:, np.newaxis]
            if use_heart_rate:
                if window_statistic == "mean":
                    activity = data["acc_norm_heart_rate"][:, np.newaxis]
                elif window_statistic == "var":
                    activity = data["imu_var"][:, np.newaxis]
                elif window_statistic == "power":
                    activity = data["imu_power"][:, np.newaxis]
                else:
                    raise NotImplementedError()

            series = np.concatenate((series, activity), axis=1)

        if use_static_features:
            #  row = static_feature_df[static_feature_df["SUBJECT_ID"] == label]
            #  static_features = row.values[:, 2:].astype(float)  # (1, 12)
            #  repeated = np.repeat(static_features, repeats=len(series), axis=0)
            #  series = np.concatenate((series, repeated), axis=1)
            activity = data["activity"]
            if use_heart_rate:
                # use the same preprocessing as for the heartrate

                encoder = OneHotEncoder(
                    categories=[list(range(0, 9))], sparse_output=False
                )
                window_size = 8
                stride = 2
                argmax_activities: list[int] = []
                activity_label_1hz = activity[::4].astype(np.int64)  # activity is 4Hz
                for i in range(0, len(activity_label_1hz) - window_size + 1, stride):
                    window = activity_label_1hz[i : i + window_size, :]
                    counts = np.bincount(window[:, 0], minlength=9)
                    argmax_activities.append(int(np.argmax(counts)))

                processed_activity = np.array(argmax_activities)[:, np.newaxis]
                assert len(processed_activity) == len(series)

                processed_activity_onehot = encoder.fit_transform(processed_activity)
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


class DaLiADataset(Dataset[tuple[Tensor, Tensor]]):
    def __init__(
        self,
        path: str,
        participants: list[int],
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        use_heart_rate: bool = False,
        look_back_window: int = 32,
        prediction_window: int = 10,
        target_channel_dim: int = 1,
        window_statistic: str = "mean",
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.participants = participants
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.target_channel_dim = target_channel_dim
        self.data, [self.mean, self.std, self.min, self.max] = dalia_load_data(
            path,
            participants,
            use_heart_rate,
            use_dynamic_features,
            use_static_features,
            window_statistic=window_statistic,
        )

        self.lengths = [len(series) - self.window_length + 1 for series in self.data]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window_length)]
        look_back_window = torch.from_numpy(window[: self.look_back_window, :])
        prediction_window = torch.from_numpy(window[self.look_back_window :, :])

        return look_back_window.float(), prediction_window.float()


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
        if stage == "fit":
            self.train_dataset = DaLiADataset(
                self.data_dir,
                self.train_participants,
                self.use_dynamic_features,
                self.use_static_features,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.target_channel_dim,
                self.window_statistic,
            )
            self.val_dataset = DaLiADataset(
                self.data_dir,
                self.val_participants,
                self.use_dynamic_features,
                self.use_static_features,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.target_channel_dim,
                self.window_statistic,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                self.data_dir,
                self.test_participants,
                self.use_dynamic_features,
                self.use_static_features,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.target_channel_dim,
                self.window_statistic,
            )
