import numpy as np
import torch

from pathlib import Path
from scipy.io import loadmat
from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder
from typing import Tuple

from src.datasets.utils import BaseDataModule


def usc_load_data(datadir: str, participants: list[int], use_activity_info: bool):
    encoder = OneHotEncoder(categories=[list(range(1, 13))], sparse_output=False)
    participant_data = []
    for participant in participants:
        participant_dir = Path(datadir) / f"Subject{participant}"
        participant_mat_paths = participant_dir.glob("*.mat")
        part_data = []
        for mat_path in participant_mat_paths:
            data = loadmat(mat_path)
            sensor_reading = data["sensor_readings"]
            if use_activity_info:
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
            part_data.append(sensor_reading)

        participant_data.append(part_data)

    return participant_data


class USCDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        look_back_window: int = 10,
        prediction_window: int = 5,
        participants: list[int] = [1, 2, 3, 4, 5, 6, 7, 8],
        use_activity_info: bool = False,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.base_channel_dim = 6  # 3 channels for acceleration and 3 channels for gyro
        self.participants = participants

        self.part_cum_sum = []
        self.participant_data = usc_load_data(data_dir, participants, use_activity_info)
        for participant_idx in range(len(self.participant_data)):
            lengths = [
                len(series) - self.window_length + 1
                for series in self.participant_data[participant_idx]
            ]
            self.part_cum_sum.append(np.cumsum([0] + lengths))
        outer_lengths = [cumsum[-1] for cumsum in self.part_cum_sum]
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
            (series_pos + self.look_back_window) : (series_pos + self.window_length),
            : self.base_channel_dim,
        ]

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
        use_activity_info: bool = False,
        num_workers: int = 0,
        freq: int = 100,
        name: str = "usc",
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

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_activity_info,
            )
            self.val_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = USCDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_activity_info,
            )


if __name__ == "__main__":
    module = USCDataModule("C:/Users/cleme/ETH/Master/Thesis/data/USC/USC-HAD/")
    module.setup("fit")

    t_d = module.train_dataloader()

    tensor = next(iter(t_d))

    import pdb

    pdb.set_trace()
