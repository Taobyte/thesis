import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.preprocessing import OneHotEncoder

from src.datasets.utils import BaseDataModule


def ucihar_load_data(datadir: str, participants: list[int], use_static_features: bool):
    data = np.load(datadir + "ucihar_preprocessed.npz")

    encoder = OneHotEncoder(categories=[list(range(1, 7))], sparse_output=False)

    if participants:
        X_train_val = data["X_train"]
        train_val_participants = data["train_val_subjects"]
        filter_vector = np.isin(train_val_participants, np.array(participants))
        series = X_train_val[filter_vector]
        if use_static_features:
            y_train_val = encoder.fit_transform(
                data["y_train"][filter_vector].astype(int).reshape(-1, 1)
            )
            y_expanded = np.repeat(y_train_val[:, np.newaxis, :], repeats=128, axis=1)
            series = np.concatenate((series, y_expanded), axis=-1)
    else:
        X_test = data["X_test"]
        series = X_test
        if use_static_features:
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
        use_static_features: bool = False,
        target_channel_dim: int = 9,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        assert self.window <= 128

        self.target_channel_dim = target_channel_dim
        self.use_static_features = use_static_features

        self.X = ucihar_load_data(datadir, participants, use_static_features)

    def __len__(self) -> int:
        return len(self.X) * (128 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        row_idx = idx // (128 - self.window + 1)
        window_pos = idx % (128 - self.window + 1)
        window = self.X[row_idx, window_pos : window_pos + self.window]
        look_back_window = torch.from_numpy(window[: self.look_back_window]).float()
        prediction_window = torch.from_numpy(
            window[self.look_back_window :, : self.target_channel_dim]
        ).float()
        return look_back_window, prediction_window


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
        freq: int = 50,
        name: str = "ucihar",
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 6,
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

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_static_features,
            )
            self.val_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_static_features,
            )
        if stage == "test":
            self.test_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                None,
                self.use_static_features,
            )
