import pandas as pd
import numpy as np
import torch
import lightning as L

from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from scipy.io import loadmat


def chapman_load_data(datadir: str):
    data = loadmat(datadir + "chapman.mat")["whole_data"]
    df = pd.DataFrame(data)
    classes = df[1].apply(lambda x: x[0][0])

    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = stratified_split_onehot(
        df, classes
    )

    return X_train, y_train, X_val, y_val, X_test, y_test


def stratified_split_onehot(
    data: pd.DataFrame,
    labels: pd.Series,
    train_size=0.7,
    val_size=0.15,
    test_size=0.15,
    seed=42,
):
    assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Splits must sum to 1."

    X_temp, X_test, y_temp, y_test = train_test_split(
        data, labels, test_size=test_size, stratify=labels, random_state=seed
    )

    val_ratio = val_size / (train_size + val_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=seed
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
    """
    A PyTorch Dataset for windowed time series forecasting using the Chapman ECG dataset.

    Each participant has a 1000-length multivariate time series. This dataset provides
    sliding windows of (look_back_window + prediction_window) length, optionally augmented
    with repeated one-hot encoded disease class as an extra input channel.

    Parameters
    ----------
    look_back_window : int
        Number of past timesteps used for forecasting.
    prediction_window : int
        Number of future timesteps to predict.
    X : np.ndarray
        The ECG time series data of shape (n_participants, 1000, channels).
    disease : np.ndarray
        One-hot encoded disease labels of shape (n_participants, num_classes).
    use_disease : bool, optional
        Whether to include the disease label as a repeated conditioning input.
    """

    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        X: np.ndarray,
        disease: np.ndarray,
        use_disease: bool = False,
    ):
        self.X = X
        self.disease = disease
        self.use_disease = use_disease
        self.n_participants = len(disease)
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

    def __len__(self) -> int:
        """
        Returns the number of sliding windows in the dataset.

        Returns
        -------
        int
            Total number of samples across all participants.
        """
        return self.n_participants * (1000 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Returns a single sample consisting of a look-back window and a prediction window.

        Parameters
        ----------
        idx : int
            Index of the window sample.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Tuple of (look_back_window, prediction_window), optionally with disease one-hot vector
            concatenated as an extra channel in the look_back_window.
        """
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


class ChapmanDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule for the Chapman ECG dataset.

    Handles loading the data from .mat file, performing stratified splits into
    train/val/test sets, and preparing DataLoaders.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing `chapman.mat`.
    batch_size : int
        Number of samples per batch.
    look_back_window : int
        Length of the input window.
    prediction_window : int
        Length of the prediction window.
    use_disease: bool
        Boolean value that indicates if disease one-hot encoding is concatenated to the look_back_windows
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        use_disease: bool = False,
        freq: int = 10,  # TODO
        name: str = "chapman",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.use_disease = use_disease

        self.freq = freq
        self.name = name

        (
            self.X_train,
            self.y_train,
            self.X_val,
            self.y_val,
            self.X_test,
            self.y_test,
        ) = chapman_load_data(data_dir)

    def setup(self, stage: str):
        """
        Setup datasets for a specific stage (fit/test).

        Parameters
        ----------
        stage : str
            One of {"fit", "test"} indicating which datasets to initialize.
        """
        if stage == "fit":
            self.train_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_train,
                self.y_train,
                use_disease=self.use_disease,
            )
            self.val_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_val,
                self.y_val,
                use_disease=self.use_disease,
            )
        if stage == "test":
            self.test_dataset = ChapmanDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_test,
                self.y_test,
                use_disease=self.use_disease,
            )

    def train_dataloader(self):
        """Returns DataLoader for training set."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        """Returns DataLoader for validation set."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        """Returns DataLoader for test set."""
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/Chapman/"
    X_train, y_train, X_val, y_val, X_test, y_test = chapman_load_data(datadir)
    print(X_train[0].shape)
