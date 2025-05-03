import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

from src.datasets.utils import BaseDataModule


def ucihar_load_data(datadir: str, participants: list[int], use_activity_info: bool):
    data = np.load(datadir + "ucihar_preprocessed.npz")

    encoder = OneHotEncoder(categories=[list(range(1, 7))], sparse_output=False)

    if participants:
        X_train_val = data["X_train"]
        train_val_participants = data["train_val_subjects"]
        filter_vector = np.isin(train_val_participants, np.array(participants))
        series = X_train_val[filter_vector]
        if use_activity_info:
            y_train_val = encoder.fit_transform(
                data["y_train"][filter_vector].astype(int).reshape(-1, 1)
            )
            y_expanded = np.repeat(y_train_val[:, np.newaxis, :], repeats=128, axis=1)
            series = np.concatenate((series, y_expanded), axis=-1)
    else:
        X_test = data["X_test"]
        series = X_test
        if use_activity_info:
            y_test = encoder.fit_transform(data["y_test"].astype(int).reshape(-1, 1))
            y_expanded = np.repeat(y_test[:, np.newaxis, :], repeats=128, axis=1)
            series = np.concatenate((X_test, y_expanded), axis=-1)

    return series


class UCIHARDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        participants: list[int] = [1, 2, 3],
        use_activity_info: bool = False,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        assert self.window <= 128

        self.base_channel_dim = 9
        self.use_activity_info = use_activity_info

        self.X = ucihar_load_data(datadir, participants, use_activity_info)

    def __len__(self) -> int:
        return len(self.X) * (128 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row_idx = idx // (128 - self.window + 1)
        window_pos = idx % (128 - self.window + 1)
        window = self.X[row_idx, window_pos : window_pos + self.window]
        x = torch.from_numpy(window[: self.look_back_window]).float()
        y = torch.from_numpy(
            window[self.look_back_window :, : self.base_channel_dim]
        ).float()
        return x, y


class UCIHARDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 0,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22],
        val_participants: list = [23, 25, 26, 27, 28, 29, 30],
        use_activity_info: bool = False,
        freq: int = 50,
        name: str = "ucihar",
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

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_activity_info,
            )
            self.val_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                None,
                self.use_activity_info,
            )
