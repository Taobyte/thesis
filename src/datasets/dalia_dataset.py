import numpy as np
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


def dalia_load_data(
    path: str,
    participants: list[int],
    use_heart_rate: bool,
    use_activity_info: bool,
):
    loaded_series = []
    for i in participants:
        label = "S" + str(i)
        data_path = Path(path) / (label + ".npz")
        data = np.load(data_path)
        series = data["heart_rate"][:, np.newaxis] if use_heart_rate else data["bvp"]

        if use_activity_info:
            activity = data["acc_norm_ppg"][:, np.newaxis]
            print(activity.shape)
            if use_heart_rate:
                activity = data["acc_norm_heart_rate"][:, np.newaxis]
            series = np.concatenate((series, activity), axis=1)

        loaded_series.append(series)
    return loaded_series


class DaLiADataset(Dataset):
    def __init__(
        self,
        path: Path,
        participants: list[int],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
        look_back_window: int = 32,
        prediction_window: int = 10,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.participants = participants
        self.use_activity_info = use_activity_info
        self.target_channel_dim = 1

        self.data = dalia_load_data(
            path, participants, use_heart_rate, use_activity_info
        )
        self.lengths = [len(series) - self.window_length + 1 for series in self.data]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window_length)]
        look_back_window = torch.from_numpy(window[: self.look_back_window, :])
        prediction_window = torch.from_numpy(
            window[self.look_back_window :, : self.target_channel_dim]
        )

        return look_back_window.float(), prediction_window.float()


class DaLiADataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list = [9, 10, 11],
        test_participants: list = [1, 13, 14],
        num_workers: int = 0,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.num_workers = num_workers

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = DaLiADataset(
                self.data_dir,
                self.train_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
            )
            self.val_dataset = DaLiADataset(
                self.data_dir,
                self.val_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                self.data_dir,
                self.test_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
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
