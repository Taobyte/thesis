import numpy as np
import pandas as pd
import torch
import lightning as L

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple

from src.datasets.utils import BaseDataModule


def stratified_participant_split(
    df: pd.DataFrame, test_size=0.2, val_size=0.25, seed=42
):
    df["strata"] = df["age"].astype(str) + "_" + df["sex"].astype(str)

    train_val, test = train_test_split(
        df, test_size=test_size, stratify=df["strata"], random_state=seed
    )

    train, val = train_test_split(
        train_val, test_size=val_size, stratify=train_val["strata"], random_state=seed
    )

    return (
        train.reset_index(drop=True)["pid"],
        val.reset_index(drop=True)["pid"],
        test.reset_index(drop=True)["pid"],
    )


class Capture24Dataset(Dataset):
    def __init__(
        self,
        datadir: str,
        participants: list[str],
        look_back_window: int = 10,
        prediction_window: int = 5,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        allowed_pids = participants

        self.file_paths = [
            file_path
            for file_path in Path(datadir).glob("P*.npy")
            if file_path.stem.split(".")[0] in allowed_pids
        ]
        self.data = [np.load(file_path, mmap_mode="r") for file_path in self.file_paths]

        self.lengths = [
            len(self.data[i]) - self.window + 1 for i in range(len(self.data))
        ]

        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        participant_idx = (
            np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        )

        pos_idx = idx - self.cumulative_lengths[participant_idx]

        window = self.data[participant_idx][pos_idx : pos_idx + self.window, :]

        look_back_window = window[: self.look_back_window, 1:4]
        prediction_window = window[self.look_back_window :, 1:4]

        return look_back_window, prediction_window


class Capture24DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        use_activity_info: bool = False,
        num_workers: int = 0,
        freq: int = 25,
        name: str = "capture24",
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            name=name,
            freq=freq,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_activity_info=use_activity_info,
        )

        metadata = pd.read_csv(data_dir + "metadata.csv")

        self.train_participants, self.val_participants, self.test_participants = (
            stratified_participant_split(metadata)
        )

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Capture24Dataset(
                self.data_dir,
                self.train_participants,
                self.look_back_window,
                self.prediction_window,
            )
            self.val_dataset = Capture24Dataset(
                self.data_dir,
                self.val_participants,
                self.look_back_window,
                self.prediction_window,
            )
        if stage == "test":
            self.test_dataset = Capture24Dataset(
                self.data_dir,
                self.test_participants,
                self.look_back_window,
                self.prediction_window,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)
