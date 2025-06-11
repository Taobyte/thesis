import numpy as np
import torch
from torch.utils.data import Dataset

from src.datasets.utils import BaseDataModule


def ieee_load_data(
    datadir: str,
    participants: list[int],
    use_heart_rate: bool,
    use_dynamic_features: bool,
):
    loaded_series = []
    for participant in participants:
        data = np.load(datadir + f"IEEE_{participant}.npz")
        ppg = data["ppg"]  # shape (W, 200, 1)
        acc = data["acc"]  # shape (W, 200, 1)
        bpm = data["bpms"]  # shape (W, 1)

        if use_heart_rate and use_dynamic_features:
            avg_acc = np.mean(acc, axis=1, keepdims=True).squeeze(-1)  # shape (W, 1)
            series = np.concatenate((bpm, avg_acc), axis=1)
        elif use_heart_rate and not use_dynamic_features:
            series = bpm
        elif not use_heart_rate and use_dynamic_features:
            series = np.concatenate((ppg, acc), axis=2)
        else:
            series = ppg

        loaded_series.append(series)

    return loaded_series


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
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.use_heart_rate = use_heart_rate
        self.use_dynamic_features = use_dynamic_features
        self.target_channel_dim = target_channel_dim

        self.data = ieee_load_data(
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
        prediction_window = torch.from_numpy(
            prediction_window[:, : self.target_channel_dim]
        ).float()

        return look_back_window, prediction_window


class IEEEDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7],
        val_participants: list[int] = [8, 9],
        test_participants: list[int] = [10, 11, 12],
        use_heart_rate: bool = False,
        freq: int = 25,
        name: str = "ieee",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 0,
        look_back_channel_dim: int = 1,
        shuffle: bool = True,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            name=name,
            freq=freq,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_dynamic_features=use_dynamic_features,
            use_static_features=use_static_features,
            target_channel_dim=target_channel_dim,
            dynamic_exogenous_variables=dynamic_exogenous_variables,
            static_exogenous_variables=static_exogenous_variables,
            look_back_channel_dim=look_back_channel_dim,
            shuffle=shuffle,
        )

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
