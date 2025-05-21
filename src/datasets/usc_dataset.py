import numpy as np
import torch

from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

from src.datasets.utils import BaseDataModule


def usc_load_data(
    datadir: str, participants: list[int], use_static_features: bool
) -> list[list[Tuple[np.ndarray, float, float, float]]]:
    encoder = OneHotEncoder(categories=[list(range(1, 13))], sparse_output=False)
    participant_data = []
    for participant in participants:
        participant_dir = Path(datadir) / f"Subject{participant}"
        participant_mat_paths = participant_dir.glob("*.mat")
        part_data = []
        for mat_path in participant_mat_paths:
            data = loadmat(mat_path)
            sensor_reading = data["sensor_readings"]
            if use_static_features:
                assert "activity_number" or "activity_numbr" in data, (
                    f"activity number not in {data.keys()}"
                )
                key = "activity_number"
                if key not in data:
                    key = "activity_numbr"
                activity_oh = encoder.fit_transform(
                    data[key].astype(int).reshape(-1, 1)
                )
                sensor_reading = np.concatenate(
                    (
                        sensor_reading,
                        np.repeat(
                            activity_oh,
                            repeats=len(sensor_reading),
                            axis=0,
                        ),
                    ),
                    axis=1,
                )
            age = data["age"].astype(float)[0]
            height = int(data["height"][0][:3])
            weight = int(data["weight"][0][:2])

            part_data.append((sensor_reading, age, height, weight))

        participant_data.append(part_data)

    return participant_data


class USCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        look_back_window: int = 10,
        prediction_window: int = 5,
        participants: list[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        use_static_features: bool = False,
        target_channel_dim: int = 6,
        look_back_channel_dim: int = 6,
        static_z_norm: list[float] = None,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.target_channel_dim = (
            target_channel_dim  # 3 channels for acceleration and 3 channels for gyro
        )
        self.look_back_channel_dim = look_back_channel_dim
        self.participants = participants

        self.part_cum_sum = []
        self.participant_data = usc_load_data(
            data_dir, participants, use_static_features
        )
        for participant_idx in range(len(self.participant_data)):
            lengths = [
                len(series) - self.window_length + 1
                for (series, _, _, _) in self.participant_data[participant_idx]
            ]
            self.part_cum_sum.append(np.cumsum([0] + lengths))
        outer_lengths = [cumsum[-1] for cumsum in self.part_cum_sum]
        self.cumulative_lengths = np.cumsum([0] + outer_lengths)
        self.total_length = self.cumulative_lengths[-1]

        self.use_static_features = use_static_features
        (
            self.age_mean,
            self.age_std,
            self.height_mean,
            self.height_std,
            self.weight_mean,
            self.weight_std,
        ) = static_z_norm

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

        mat_file, age, height, weight = self.participant_data[participant_idx][file_idx]

        window = mat_file[series_pos : series_pos + self.window_length, :]

        if self.use_static_features:
            age = (age - self.age_mean) / (self.age_std + 1e-8)
            height = (height - self.height_mean) / (self.height_std + 1e-8)
            weight = (weight - self.weight_mean) / (self.weight_std + 1e-8)
            features = np.array([age, height, weight])
            features = features[np.newaxis, :]
            repeats = np.repeat(features, self.window_length, axis=0)

            window = np.concatenate((window, repeats), axis=1)

        look_back_window = window[: self.look_back_window, : self.look_back_channel_dim]
        prediction_window = window[self.look_back_window :, : self.target_channel_dim]

        look_back_window = torch.tensor(look_back_window).float()
        prediction_window = torch.tensor(prediction_window).float()

        return look_back_window, prediction_window


class USCDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list[int] = [1, 2, 4, 5, 7, 10, 11, 14],
        val_participants: list[int] = [3, 6, 8],
        test_participants: list[int] = [9, 12, 13],
        num_workers: int = 0,
        freq: int = 100,
        name: str = "usc",
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

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        # compute static feature mean and std
        age = []
        height = []
        weight = []
        for participant in range(1, 15):
            participant_dir = Path(data_dir) / f"Subject{participant}"
            participant_mat_paths = participant_dir.glob("*.mat")
            data = loadmat(next(participant_mat_paths))
            age.append(data["age"].astype(float)[0])
            height.append(int(data["height"][0][:3]))
            weight.append(int(data["weight"][0][:2]))

        age = np.array(age)
        height = np.array(height)
        weight = np.array(weight)

        self.static_z_norm = [
            age.mean(),
            age.std(),
            height.mean(),
            height.std(),
            weight.mean(),
            weight.std(),
        ]

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
                self.static_z_norm,
            )
            self.val_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
                self.static_z_norm,
            )
        if stage == "test":
            self.test_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_static_features,
                self.target_channel_dim,
                self.look_back_channel_dim,
                self.static_z_norm,
            )
