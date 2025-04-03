import pickle
import numpy as np
import pandas as pd
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple


class DaLiADataset(Dataset):
    def __init__(
        self,
        path: Path,
        mode: str,
        use_heart_rate: bool = False,
        look_back_window: int = 320,
        prediction_window: int = 128,
        train_participants: list = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list = [9, 10, 11],
        test_participants: list = [1, 13, 14],
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        if mode == "train":
            self.participants = train_participants
        elif mode == "val":
            self.participants = val_participants
        else:
            self.participants = test_participants

        self.data = []
        self.lengths = []
        for i in self.participants:
            label = "S" + str(i)
            data_path = Path(path) / label / (label + ".pkl")
            with open(data_path, "rb") as f:
                data = pickle.load(f, encoding="latin1")

            if use_heart_rate:
                data = data["signal"]["chest"]["ECG"]
            else:
                data = data["signal"]["wrist"]["BVP"]
            self.data.append(data)
            self.lengths.append(len(data))

        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window)]
        look_back_window = torch.from_numpy(window[: self.look_back_window, 0])
        prediction_window = torch.from_numpy(window[self.look_back_window :, 0])

        return look_back_window, prediction_window


class DaLiADataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        use_heart_rate: bool = False,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list = [9, 10, 11],
        test_participants: list = [1, 13, 14],
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.use_heart_rate = use_heart_rate

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = DaLiADataset(
                self.data_dir,
                "train",
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
            )
            self.val_dataset = DaLiADataset(
                self.data_dir,
                "val",
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                self.data_dir,
                "test",
                self.use_heart_rate,
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

    modes = ["train", "val", "test"]
    use_heart_rate = [True, False]
    path = Path("C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy")
    for mode in tqdm(modes):
        for b in use_heart_rate:
            dataset = DaLiADataset(
                path,
                "train",
                use_heart_rate=False,
                look_back_window=10,
                prediction_window=5,
            )
            indices = np.random.choice(len(dataset), size=100, replace=False)
            for idx in indices:
                look_back_window, prediction_window = dataset[idx]
