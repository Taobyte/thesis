import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
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
        look_back_window: int = 100,
        prediction_window: int = 50,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 3,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 2,
        look_back_channel_dim: int = 3,
        metadata: pd.DataFrame = None,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.target_channel_dim = target_channel_dim
        self.dynamic_exogenous_variables = dynamic_exogenous_variables
        self.static_exogenous_variables = static_exogenous_variables
        self.look_back_channel_dim = look_back_channel_dim

        self.static_features = (
            metadata[metadata["pid"].isin(participants)].drop(columns=["pid"]).values
        )

        allowed_pids = participants
        self.file_paths = [
            file_path
            for file_path in Path(datadir).glob("P*.npy")
            if str(file_path.stem) in allowed_pids
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
        if self.use_dynamic_features:
            window = self.data[participant_idx][pos_idx : pos_idx + self.window, :]
        else:
            window = self.data[participant_idx][
                pos_idx : pos_idx + self.window, : self.target_channel_dim
            ]

        # load in static variables AGE & SEX
        if self.use_static_features:
            static_features_repeats = np.repeat(
                self.static_features[participant_idx][np.newaxis, :],
                repeats=len(window),
                axis=0,
            )
            window = np.concatenate((window, static_features_repeats), axis=1)

        look_back_window = torch.tensor(
            window[: self.look_back_window, : self.look_back_channel_dim]
        ).float()
        prediction_window = torch.tensor(
            window[self.look_back_window :, : self.target_channel_dim]
        ).float()

        return look_back_window, prediction_window


class Capture24DataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        num_workers: int = 0,
        freq: int = 25,
        name: str = "capture24",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 0,
        look_back_channel_dim: int = 1,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            name=name,
            freq=freq,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_dynamic_features=use_dynamic_features,
            use_static_features=use_static_features,
            target_channel_dim=target_channel_dim,
            dynamic_exogenous_variables=dynamic_exogenous_variables,
            static_exogenous_variables=static_exogenous_variables,
            look_back_channel_dim=look_back_channel_dim,
        )

        metadata = pd.read_csv(data_dir + "metadata.csv")

        train_participants, val_participants, test_participants = (
            stratified_participant_split(metadata)
        )

        onehot = pd.get_dummies(metadata["age"], prefix="age") * 1
        metadata = pd.concat((metadata, onehot), axis=1)
        metadata["sex"] = metadata["sex"].apply(lambda x: 0 if x == "F" else 1)
        metadata = metadata.sort_values("pid")
        metadata = metadata.drop(columns=["age", "strata"])
        self.metadata = metadata

        self.train_participants = train_participants.tolist()
        self.val_participants = val_participants.tolist()
        self.test_participants = test_participants.tolist()

        train_set = set(self.train_participants)
        val_set = set(self.val_participants)
        test_set = set(self.test_participants)

        assert train_set.isdisjoint(val_set), "Train and Val sets are not disjoint"
        assert train_set.isdisjoint(test_set), "Train and Test sets are not disjoint"
        assert val_set.isdisjoint(test_set), "Val and Test sets are not disjoint"

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = Capture24Dataset(
                self.data_dir,
                self.train_participants,
                self.look_back_window,
                self.prediction_window,
                self.use_dynamic_features,
                self.use_static_features,
                self.target_channel_dim,
                self.dynamic_exogenous_variables,
                self.static_exogenous_variables,
                self.look_back_channel_dim,
                self.metadata,
            )
            self.val_dataset = Capture24Dataset(
                self.data_dir,
                self.val_participants,
                self.look_back_window,
                self.prediction_window,
                self.use_dynamic_features,
                self.use_static_features,
                self.target_channel_dim,
                self.dynamic_exogenous_variables,
                self.static_exogenous_variables,
                self.look_back_channel_dim,
                self.metadata,
            )
        if stage == "test":
            self.test_dataset = Capture24Dataset(
                self.data_dir,
                self.test_participants,
                self.look_back_window,
                self.prediction_window,
                self.use_dynamic_features,
                self.use_static_features,
                self.target_channel_dim,
                self.dynamic_exogenous_variables,
                self.static_exogenous_variables,
                self.look_back_channel_dim,
                self.metadata,
            )
