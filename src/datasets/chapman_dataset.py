import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat

from src.datasets.utils import BaseDataModule


def chapman_load_data(datadir: str, random_state: int = 42):
    data = loadmat(datadir + "chapman.mat")["whole_data"]

    df = pd.DataFrame(data)
    classes = df[1].apply(lambda x: x[0][0])

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = stratified_split_onehot(
        df, classes, random_state=random_state
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def stratified_split_onehot(
    data: pd.DataFrame,
    labels: pd.Series,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    random_state=42,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1."

    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=random_state
    )

    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )

    encoder = OneHotEncoder(sparse_output=False)
    y_train_oh = encoder.fit_transform(y_train.values.reshape(-1, 1))
    y_val_oh = encoder.transform(y_val.values.reshape(-1, 1))
    y_test_oh = encoder.transform(y_test.values.reshape(-1, 1))

    return (
        (X_train[0].values, y_train_oh),
        (X_val[0].values, y_val_oh),
        (X_test[0].values, y_test_oh),
        encoder,
    )


class ChapmanDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        X: np.ndarray,
        disease: np.ndarray,
        use_static_features: bool = False,
    ):
        self.X = X
        self.disease = disease
        self.use_disease = use_static_features
        self.n_participants = len(disease)
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        assert self.window <= 1000

    def __len__(self) -> int:
        return self.n_participants * (1000 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        participant_idx = idx // (1000 - self.window + 1)
        window_pos = idx % (1000 - self.window + 1)

        look_back_window = self.X[participant_idx][
            window_pos : window_pos + self.look_back_window, :
        ]
        prediction_window = self.X[participant_idx][
            window_pos + self.look_back_window : window_pos + self.window, :
        ]

        if self.use_disease:
            disease_vec = np.repeat(
                self.disease[participant_idx][np.newaxis, :],
                self.look_back_window,
                axis=0,
            )
            look_back_window = np.concatenate((look_back_window, disease_vec), axis=1)

        return torch.from_numpy(look_back_window).float(), torch.from_numpy(
            prediction_window
        ).float()


class ChapmanDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        freq: int = 100,
        name: str = "chapman",
        num_workers: int = 0,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 4,
        dynamic_exogenous_variables: int = 0,
        static_exogenous_variables: int = 4,
        look_back_channel_dim: int = 4,
        random_state: int = 42,
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

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = chapman_load_data(data_dir)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_train,
                self.y_train,
                use_static_features=self.use_static_features,
            )
            self.val_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_val,
                self.y_val,
                use_static_features=self.use_static_features,
            )
        if stage == "test":
            self.test_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_test,
                self.y_test,
                use_static_features=self.use_static_features,
            )
