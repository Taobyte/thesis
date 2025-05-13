import pandas as pd
import numpy as np
import pickle

from torch.utils.data import Dataset
from src.datasets.utils import BaseDataModule
from sklearn.model_selection import train_test_split
from typing import Tuple


def get_train_test_split(
    df: pd.Dataframe, ids: list[str]
) -> Tuple[list[str], list[str], list[str]]:
    # Drop rows with missing values in Gender or age
    df_strat = df.dropna(subset=["Gender", "age", "recordId"]).copy()

    # Optional: bin age for better stratification (e.g., 10-year bins)
    df_strat["age_bin"] = pd.cut(
        df_strat["age"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False
    )

    # Combine Gender and age_bin for stratification
    df_strat["strat_col"] = (
        df_strat["Gender"].astype(str) + "_" + df_strat["age_bin"].astype(str)
    )

    # First split: Train+Val and Test (e.g., 80% / 20%)
    trainval_ids, test_ids = train_test_split(
        df_strat["recordId"],
        test_size=0.2,
        random_state=42,
        stratify=df_strat["strat_col"],
    )

    # Prepare DataFrame for second split
    trainval_df = df_strat[df_strat["recordId"].isin(trainval_ids)].copy()

    # Second split: Train and Validation (e.g., 80% / 20% of trainval)
    train_ids, val_ids = train_test_split(
        trainval_df["recordId"],
        test_size=0.25,  # 0.25 * 0.8 = 0.2 of total
        random_state=42,
        stratify=trainval_df["strat_col"],
    )

    return train_ids, val_ids, test_ids


class MHC6MWTDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        timeseries: list[np.ndarray],
        use_activity_info: bool = False,
    ):
        self.datadir = datadir
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.use_activity_info = use_activity_info
        self.base_channel_dim = 1

        self.cumsum = np.cumsum(
            [
                len(arr) - self.window_length + 1
                for arr in timeseries
                if len(arr) >= self.window_length
            ]
        )

        self.total_length = self.cumsum[-1]

    def __len__(self):
        return self.total_length

    def __getitem__(self, index):
        return super().__getitem__(index)


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

        with open(data_dir + "/mhc6mwt.pkl", "rb") as f:
            data = pickle.load(f)

        ids = list(data.keys())
        summary_df = pd.read_csv(data_dir + "/summary.parquet")
        train_ids, val_ids, test_ids = get_train_test_split(
            summary_df, ids
        )  # we only consider keys that are in the data

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
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_arrays,
                self.use_activity_info,
            )
            self.val_dataset = MHC6MWTDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_arrays,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = MHC6MWTDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_arrays,
                self.use_activity_info,
            )
