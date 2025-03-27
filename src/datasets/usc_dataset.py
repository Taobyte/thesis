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
        look_back_window: int = 320,
        prediction_window: int = 128,
        subjects: list = [1, 2, 3, 4, 5, 6, 7, 8, 9],
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        self.data_dir = Path(data_dir)
        self.subjects = subjects

        self.sub_mats = []
        for i in range(len(self.subjects)):
            self.sub_mats.append(list(self.data_dir.glob(f"Subject{i + 1}/*.mat")))

    def __len__(self) -> int:
        return len(self.subjects) * 60

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        subject = index // 60
        mat_file_idx = index - subject * 60
        mat_file = loadmat(self.sub_mats[subject][mat_file_idx])["sensor_readings"]

        acc = mat_file[:, :3]
        ger = mat_file[:, 3:]

        acc_norm = np.sqrt(np.sum(np.square(acc), axis=1))
        ger_norm = np.sqrt(np.sum(np.square(ger), axis=1))

        # random sampling
        n_windows = len(acc_norm) - self.window + 1
        start = torch.randint(n_windows, (1,)).item()
        x = acc_norm[start : start + self.look_back_window]
        y = acc_norm[start + self.look_back_window : start + self.window]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


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
    usc_path = Path("C:/Users/cleme/ETH/Master/Thesis/data/USC/USC-HAD")
    dataset = USCDataset(usc_path)
    for i in range(len(dataset)):
        x, y = dataset[i]
