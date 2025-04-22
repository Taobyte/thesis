import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset


def ieee_load_data(
    datadir: str, participants: list[int], use_heart_rate: bool, use_activity_info: bool
):
    loaded_series = []
    for participant in participants:
        data = np.load(datadir + f"IEEE_{participant}.npz")
        ppg = data["ppg"]  # shape (W, 200, 1)
        acc = data["acc"]  # shape (W, 200, 1)
        bpm = data["bpms"]  # shape (W, 1)

        if use_heart_rate and use_activity_info:
            avg_acc = np.mean(acc, axis=1, keepdims=True).squeeze(-1)  # shape (W, 1)
            series = np.concatenate((bpm, avg_acc), axis=1)
        elif use_heart_rate and not use_activity_info:
            series = bpm
        elif not use_heart_rate and use_activity_info:
            series = np.concatenate((ppg, acc), axis=2)
        else:
            series = ppg

        loaded_series.append(series)

    combined_series = np.concatenate(
        [arr.reshape(-1, 1) for arr in loaded_series], axis=0
    )

    global_mean = np.mean(combined_series, axis=0, keepdims=True)[np.newaxis, :, :]
    global_std = np.std(combined_series, axis=0, keepdims=True)[np.newaxis, :, :]

    return loaded_series, global_mean, global_std


class IEEEDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        participants: list[int],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
    ):
        self.datadir = datadir
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info
        self.base_channel_dim = 1

        self.data, self.global_mean, self.global_std = ieee_load_data(
            datadir, participants, use_heart_rate, use_activity_info
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
            prediction_window[:, : self.base_channel_dim]
        ).float()

        return look_back_window, prediction_window


class IEEEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        val_participants: list[int] = [15, 16, 17, 18],
        test_participants: list[int] = [19, 20, 21, 22],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )
            self.val_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )
