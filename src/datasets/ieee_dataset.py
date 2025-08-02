import numpy as np
import torch

from torch.utils.data import Dataset
from typing import Tuple, Any

from src.datasets.utils import BaseDataModule


def ieee_load_data(
    datadir: str,
    participants: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loaded_series = []
    for participant in participants:
        data = np.load(datadir + f"IEEE_{participant}.npz")
        ppg = data["ppg"]  # shape (W, 200, 1)
        acc = data["acc"]  # shape (W, 200, 1)
        bpm = data["bpms"]  # shape (W, 1)

        if use_heart_rate and use_dynamic_features:
            imu_mean = np.mean(acc, axis=1, keepdims=True).squeeze(-1)  # shape (W, 1)
            imu_var = np.var(acc, axis=1, keepdims=True).squeeze(-1)
            imu_power = np.mean(acc**2, axis=1, keepdims=True).squeeze(-1)
            imu_energy = np.sum(acc**2, axis=1, keepdims=True).squeeze(-1)
            series = np.concatenate((bpm, imu_mean), axis=1)
        elif use_heart_rate and not use_dynamic_features:
            series = bpm
        elif not use_heart_rate and use_dynamic_features:
            series = np.concatenate((ppg, acc), axis=2)
        else:
            series = ppg

        loaded_series.append(series)

    combined = np.concatenate(loaded_series, axis=0)
    mean = np.mean(combined, axis=0)
    std = np.std(combined, axis=0)
    min = np.min(combined, axis=0)
    max = np.max(combined, axis=0)

    return loaded_series, mean, std, min, max


class IEEEDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        participants: list[int],
        use_heart_rate: bool = False,
        use_dynamic_features: bool = False,
        target_channel_dim: int = 1,
    ):
        self.datadir = datadir
        self.participants = participants
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.use_heart_rate = use_heart_rate
        self.use_dynamic_features = use_dynamic_features
        self.target_channel_dim = target_channel_dim

        self.data, self.mean, self.std, self.min, self.max = ieee_load_data(
            datadir, participants, use_heart_rate, use_dynamic_features
        )

        assert self.window_length <= 200, (
            f"window_length: {self.window_length}: IEEE for PPG contains only time series of length 200!"
        )

        self.series_per_participant = [len(series) for series in self.data]
        self.n_windows_per_participant = [
            n_series * (200 - self.window_length + 1)
            for n_series in self.series_per_participant
        ]

        if use_heart_rate:
            self.cumulative_sum = np.cumsum(
                [0]
                + [
                    (l - self.window_length + 1)
                    for l in self.series_per_participant
                    if l >= self.window_length
                ]
            )
            self.data = [d for d in self.data if len(d) >= self.window_length]

        else:
            self.cumulative_sum = np.cumsum([0] + self.n_windows_per_participant)
        self.total_length = self.cumulative_sum[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        participant_idx = np.searchsorted(self.cumulative_sum, idx, side="right") - 1
        index = idx - self.cumulative_sum[participant_idx]

        if self.use_heart_rate:
            data = self.data[participant_idx]
            window_pos = index
        else:
            serie_idx = index // (200 - self.window_length + 1)
            window_pos = index % (200 - self.window_length + 1)
            data = self.data[participant_idx][serie_idx]

        look_back_window = data[window_pos : window_pos + self.look_back_window]
        prediction_window = data[
            window_pos + self.look_back_window : window_pos + self.window_length
        ]

        look_back_window = torch.from_numpy(look_back_window).float()
        prediction_window = torch.from_numpy(prediction_window[:, :]).float()

        return look_back_window, prediction_window


class IEEEDataModule(BaseDataModule):
    def __init__(
        self,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7],
        val_participants: list[int] = [8, 9],
        test_participants: list[int] = [10, 11, 12],
        use_heart_rate: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_heart_rate,
                self.use_dynamic_features,
                self.target_channel_dim,
            )
            self.val_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_heart_rate,
                self.use_dynamic_features,
                self.target_channel_dim,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_heart_rate,
                self.use_dynamic_features,
                self.target_channel_dim,
            )
