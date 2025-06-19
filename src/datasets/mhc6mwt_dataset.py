import pandas as pd
import numpy as np
import torch
import pickle

from torch.utils.data import Dataset
from src.datasets.utils import BaseDataModule
from sklearn.model_selection import train_test_split
from typing import Tuple


def get_train_test_split(
    df: pd.DataFrame, ids: list[str], random_state: int
) -> Tuple[list[str], list[str], list[str]]:
    df_strat = df.copy()
    df_strat = df_strat[df_strat["recordId"].isin(ids)]

    df_strat["age_bin"] = pd.cut(
        df_strat["age"], bins=[0, 20, 30, 40, 50, 60, 70, 80, 100], labels=False
    )

    trainval_ids, test_ids = train_test_split(
        df_strat["recordId"],
        test_size=0.2,
        random_state=random_state,
        stratify=df_strat["age_bin"],
    )

    trainval_df = df_strat[df_strat["recordId"].isin(trainval_ids)].copy()

    train_ids, val_ids = train_test_split(
        trainval_df["recordId"],
        test_size=0.25,
        random_state=random_state,
        stratify=trainval_df["age_bin"],
    )

    return train_ids, val_ids, test_ids


class MHC6MWTDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        timeseries: list[np.ndarray],
        static_features: np.ndarray,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        look_back_channel_dim: int = 1,
    ):
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window = look_back_window + prediction_window
        self.use_static_features = use_static_features
        self.target_channel_dim = target_channel_dim
        self.look_back_channel_dim = look_back_channel_dim

        # filter out all series that are not long enough
        self.static_features = [
            static_features
            for i in range(len(static_features))
            if len(timeseries[i]) >= self.window
        ]
        self.data = [
            timeseries[i]
            for i in range(len(timeseries))
            if len(timeseries[i]) >= self.window
        ]

        combined = np.concatenate(self.data, axis=0)
        self.mean = np.mean(combined, axis=0)
        self.std = np.std(combined, axis=0)

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
        """
        if self.use_static_features:
            repeated = np.repeat(
                self.static_features[participant_idx][np.newaxis, :],
                self.window,
                axis=0,
            )
            window = np.concatenate((window, repeated), axis=1)
        """

        look_back_window = torch.tensor(
            window[: self.look_back_window, : self.look_back_channel_dim]
        ).float()
        prediction_window = torch.tensor(
            window[self.look_back_window :, : self.target_channel_dim]
        ).float()

        return look_back_window, prediction_window


class MHC6MWTDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        random_state: int = 42,
        use_heart_rate: bool = True,
        **kwargs,
    ):
        super().__init__(data_dir=data_dir, **kwargs)

        self.use_heart_rate = use_heart_rate

        with open(data_dir + "mhc6mwt.pkl", "rb") as f:
            data = pickle.load(f)

        # mean, scale  = data["z_norm"]

        ids = list(data.keys())
        summary = pd.read_parquet(data_dir + "summary_table.parquet")

        train_ids, val_ids, test_ids = get_train_test_split(
            summary, ids, random_state
        )  # we only consider keys that are in the data

        # static features
        summary["age"] = (summary["age"] - summary["age"].mean()) / (
            summary["age"].std() + 1e-8
        )
        phys_activity_one_hot = (
            pd.get_dummies(summary["phys_activity"], prefix="phys_act", dummy_na=True)
            * 1
        )
        sex_one_hot = (
            pd.get_dummies(
                summary["BiologicalSex"],
                prefix="sex",
                dummy_na=True,
            )
            * 1
        )
        self.static_features = pd.concat(
            (summary[["recordId", "age"]], phys_activity_one_hot, sex_one_hot),
            axis=1,
        )

        train_set = set(train_ids)
        val_set = set(val_ids)
        test_set = set(test_ids)

        self.train_ids = train_ids
        self.val_ids = val_ids
        self.test_ids = test_ids

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

    def get_static_features(self, ids: list[str]) -> np.ndarray:
        df = self.static_features.copy()
        df = df[df["recordId"].isin(ids)]
        df = df.sort_values("recordId")
        df = df.drop(columns=["recordId"])
        return df.values

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.train_arrays,
                self.get_static_features(self.train_ids),
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
            )
            self.val_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.val_arrays,
                self.get_static_features(self.val_ids),
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
            )
        if stage == "test":
            self.test_dataset = MHC6MWTDataset(
                self.look_back_window,
                self.prediction_window,
                self.test_arrays,
                self.get_static_features(self.test_ids),
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
            )
