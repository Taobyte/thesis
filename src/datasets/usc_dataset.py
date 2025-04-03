import numpy as np
import torch
import lightning as L

from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader
from typing import Tuple


class USCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        mode: str,
        look_back_window: int = 10,
        prediction_window: int = 5,
        train_participants: list = [1, 2, 3, 4, 5, 6, 7, 8],
        val_participants: list = [9, 10, 11],
        test_participants: list = [12, 13, 14],
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        self.data_dir = Path(data_dir)
        if mode == "train":
            self.participants = train_participants
        elif mode == "val":
            self.participants = val_participants
        else:
            self.participants = test_participants

        self.part_cum_sum = []
        self.participant_data = []
        outer_lengths = []
        for participant in self.participants:
            participant_dir = self.data_dir / f"Subject{participant}"
            participant_mat_paths = participant_dir.glob("*.mat")
            part_data = []
            lengths = []
            for mat_path in participant_mat_paths:
                mat_file = loadmat(mat_path)["sensor_readings"]
                mat_length = len(mat_file) - self.window + 1
                part_data.append(mat_file)
                lengths.append(mat_length)

            cumsum = np.cumsum([0] + lengths)
            self.part_cum_sum.append(cumsum)
            total_length = cumsum[-1]
            outer_lengths.append(total_length)
            self.participant_data.append(part_data)

        self.cumulative_lengths = np.cumsum([0] + outer_lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        participant_idx = (
            np.searchsorted(self.cumulative_lengths, index, side="right") - 1
        )
        idx = index - self.cumulative_lengths[participant_idx]

        file_idx = (
            np.searchsorted(self.part_cum_sum[participant_idx], idx, side="right") - 1
        )

        series_pos = idx - self.part_cum_sum[participant_idx][file_idx]

        mat_file = self.participant_data[participant_idx][file_idx]

        look_back_window = mat_file[series_pos : series_pos + self.look_back_window, :]
        prediction_window = mat_file[
            (series_pos + self.look_back_window) : (series_pos + self.window), :
        ]

        return look_back_window, prediction_window


class USCDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.train_participants = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.val_participants = [10, 11, 12]
        self.test_participants = [13, 14, 15]

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
            )
            self.val_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    from tqdm import tqdm

    usc_path = Path("C:/Users/cleme/ETH/Master/Thesis/data/USC/USC-HAD")
    dataset = USCDataset(usc_path, "train")
    for i in tqdm(range(len(dataset))):
        x, y = dataset[i]
