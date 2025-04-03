import pandas as pd
import numpy as np
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class WildPPGDataset(Dataset):
    def __init__(
        self,
        path: Path,
        look_back_window: int = 320,
        prediction_window: int = 128,
        cases: list = [1, 2, 3, 4, 5, 6, 7, 8, 9],
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        self.arrays = []
        self.lengths = [len(arr) - self.window + 1 for arr in self.arrays]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.arrays[file_idx][index : (index + self.window)]
        x = torch.Tensor(window.iloc[: self.look_back_window, 0].values)
        y = torch.Tensor(window.iloc[self.look_back_window :, 0].values)

        return x, y


class WildPPGDataModule(L.LightningDataModule):
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
            self.train_dataset = WildPPGDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
            )
            self.val_dataset = WildPPGDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = WildPPGDataset(
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

    path = Path("C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/WildPPG/data/")
    dataset = WildPPGDataset(path)
    for i in tqdm(range(len(dataset))):
        x, y = dataset[i]
        break
