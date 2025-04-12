import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat


class IEEEDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        participants: list[int],
        use_heart_rate: bool = False,
    ):
        self.datadir = datadir
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window = self.look_back_window + self.predicition_window
        self.use_heart_rate = use_heart_rate

        data = loadmat(self.datadir + "IEEE_Big.mat")["whole_dataset"]

        self.series_per_participant = [
            len(data[i][0])
            for i in range(len(participants))
            if len(data[i][0])
            >= self.window  # ensures that length of the series is at least as long as a window
        ]
        self.n_windows_per_participant = [
            n_series * (200 - self.window + 1)
            for n_series in self.series_per_participant
        ]

        if use_heart_rate:
            self.data = [data[i][1].squeeze(-1) for i in range(len(participants))]
            self.cumulative_sum = np.cumsum(
                [0] + [(l - self.window + 1) for l in self.series_per_participant]
            )
            self.total_length = self.cumulative_sum[-1]
        else:
            self.data = [data[i][0] for i in range(len(participants))]
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
            serie_idx = index // (200 - self.window + 1)
            window_pos = index % (200 - self.window + 1)
            data = self.data[participant_idx][serie_idx]

        look_back_window = data[window_pos : window_pos + self.look_back_window]
        prediction_window = data[
            window_pos + self.look_back_window : window_pos + self.window
        ]

        look_back_window = torch.from_numpy(look_back_window[:, np.newaxis]).float()
        prediction_window = torch.from_numpy(prediction_window[:, np.newaxis]).float()

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

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_heart_rate,
            )
            self.val_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_heart_rate,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_heart_rate,
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
