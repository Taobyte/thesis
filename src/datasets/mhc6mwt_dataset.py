import pandas as pd
import numpy as np
import torch
import pickle

from torch.utils.data import Dataset
from src.datasets.utils import BaseDataModule
from sklearn.model_selection import train_test_split
from typing import Tuple


def get_train_test_split(
    df: pd.DataFrame, ids: list[str]
) -> Tuple[list[str], list[str], list[str]]:
    df_strat = df.copy()
    df_strat = df_strat[df_strat["recordId"].isin(ids)]

    df_strat["age_bin"] = pd.cut(
        df_strat["age"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False
    )

    trainval_ids, test_ids = train_test_split(
        df_strat["recordId"],
        test_size=0.2,
        random_state=42,
        stratify=df_strat["age_bin"],
    )

    trainval_df = df_strat[df_strat["recordId"].isin(trainval_ids)].copy()

    train_ids, val_ids = train_test_split(
        trainval_df["recordId"],
        test_size=0.25,
        random_state=42,
        stratify=trainval_df["age_bin"],
    )

    return train_ids, val_ids, test_ids


class MHC6MWTDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        timeseries: list[np.ndarray],
        use_activity_info: bool = False,
    ):
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window = look_back_window + prediction_window
        self.use_activity_info = use_activity_info
        self.base_channel_dim = 1
        self.input_channels = 3 if use_activity_info else 1

        self.data = timeseries

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

        look_back_window = torch.tensor(
            window[: self.look_back_window, : self.input_channels]
        ).float()
        prediction_window = torch.tensor(
            window[self.look_back_window :, : self.base_channel_dim]
        ).float()

        return look_back_window, prediction_window


class MHC6MWTDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        look_back_window: int = 5,
        prediction_window: int = 3,
        use_activity_info: bool = False,
        freq: int = 25,
        name: str = "mhc6mwt",
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

        with open(data_dir + "mhc6mwt.pkl", "rb") as f:
            data = pickle.load(f)

        ids = list(data.keys())
        summary_df = pd.read_parquet(data_dir + "summary_table.parquet")
        train_ids, val_ids, test_ids = get_train_test_split(
            summary_df, ids
        )  # we only consider keys that are in the data

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        assert train_set.isdisjoint(val_set), "Train and Val sets are not disjoint"
        assert train_set.isdisjoint(test_set), "Train and Test sets are not disjoint"
        assert val_set.isdisjoint(test_set), "Val and Test sets are not disjoint"

        self.train_arrays = [
            data[record_id] for record_id in train_ids if record_id in data
        ]
        self.val_arrays = [
            data[record_id] for record_id in val_ids if record_id in data
        ]
        self.test_arrays = [
            data[record_id] for record_id in test_ids if record_id in data
        ]

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.train_arrays,
                self.use_activity_info,
            )
            self.val_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.val_arrays,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.test_arrays,
                self.use_activity_info,
            )
