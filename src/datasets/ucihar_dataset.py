import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import IterableDataset
from torch.utils.data import DataLoader, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.transform import (
    InstanceSplitter,
    InstanceSampler,
    ExpectedNumInstanceSampler,
    TransformedDataset,
    Chain,
    Transformation,
)


class TransformedIterableDataset(IterableDataset):
    """
    A transformed iterable dataset that applies a transformation pipeline on-the-fly.

    Parameters:
    ----------
    dataset : Dataset
        The original dataset to transform.
    transform : Transformation
        The transformation pipeline to apply.
    is_train : bool, optional, default=True
        Whether the dataset is used for training.
    """

    def __init__(
        self, dataset: Dataset, transform: Transformation, is_train: bool = True
    ):
        super().__init__()

        self.transformed_dataset = TransformedDataset(
            dataset,
            transform,
            is_train=is_train,
        )

    def __iter__(self):
        return iter(self.transformed_dataset)


def create_dataloader(
    stage: str,
    datadir: str,
    start_time: str,
    freq: str,
    sampler: InstanceSampler,
    look_back_window: int,
    prediction_window: int,
) -> DataLoader:
    start_time = pd.Timestamp(start_time)
    datadir += "train/"
    subject_train = np.loadtxt(datadir + "subject_train.txt")[:, np.newaxis]
    x_train = np.loadtxt(datadir + "X_train.txt")
    y_train = np.loadtxt(datadir + "y_train.txt")[:, np.newaxis]

    combined = np.concatenate([subject_train, y_train, x_train], axis=1)
    df = pd.DataFrame(combined)
    df.columns = ["subject", "action"] + [
        "feature" + str(i) for i in range(x_train.shape[1])
    ]

    train_time_series = []

    for subject in df["subject"].unique():
        target = df[df["subject"] == subject].iloc[:, 2].values.T
        action_feature = df[df["subject"] == subject]["action"].values[:, np.newaxis].T

        data_entry = {
            FieldName.TARGET: target,
            FieldName.START: start_time,
            FieldName.ITEM_ID: subject,
            FieldName.FEAT_DYNAMIC_CAT: action_feature,
            FieldName.OBSERVED_VALUES: target,
        }
        train_time_series.append(data_entry)

    train_ds = ListDataset(
        data_iter=[series.copy() for series in train_time_series], freq=freq
    )

    instance_splitter = InstanceSplitter(
        target_field=FieldName.TARGET,
        is_pad_field=FieldName.IS_PAD,
        start_field=FieldName.START,
        forecast_start_field=FieldName.FORECAST_START,
        instance_sampler=ExpectedNumInstanceSampler(  # Sample strategy
            num_instances=1.0,  # Try to sample 1 instance per time step on average
            min_future=prediction_window,
        ),
        past_length=look_back_window,
        future_length=prediction_window,
        time_series_fields=[
            FieldName.OBSERVED_VALUES,
            FieldName.FEAT_DYNAMIC_CAT,
        ],
        dummy_value=0.0,
    )
    transformation = Chain([instance_splitter])

    transformed_dataset = TransformedIterableDataset(
        train_ds, transformation, is_train=True
    )
    """
    train_dataloader = DataLoader(
        transformed_dataset, batch_size=32, num_workers=0, collate_fn=batchify
    )
    """
    return transformed_dataset


class UCIHARDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        mode: str,
        look_back_window: int,
        prediction_window: int,
        train_participants: list = [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22],
        val_participants: list = [4, 5, 6],
        freq: str = "20L",
        start_time: str = "2000-01-01 00:00:00",
        sampler: InstanceSampler = None,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        datadir += mode + "/"

        subject = np.loadtxt(datadir + f"subject_{mode}.txt")[:, np.newaxis]
        x = np.loadtxt(datadir + f"X_{mode}.txt")
        y = np.loadtxt(datadir + f"y_{mode}.txt")[:, np.newaxis]
        combined = np.concatenate([subject, y, x], axis=1)

        self.data = []
        self.lengths = []
        for train_participant in train_participants:
            par_data = combined[np.where(combined[:, 0] == train_participant)]
            length = len(par_data) - self.window + 1
            self.lengths.append(length)
            self.data.append(par_data)

        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window), 2:]
        x = torch.from_numpy(window[: self.look_back_window])
        y = torch.from_numpy(window[self.look_back_window :])
        return x, y


class UCIHARDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.train_participants = [1, 2, 3, 4, 5, 6, 7, 8, 9]
        self.val_participants = [10, 11, 12]
        self.test_participants = [13, 14, 15]

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
            )
            self.val_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = UCIHARDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    datadir = (
        "C:/Users/cleme/ETH/Master/Thesis/data/UCIHAR/UCI HAR Dataset/UCI HAR Dataset/"
    )

    dataset = UCIHARDataset(datadir, "train", 10, 5)

    for i in range(len(dataset)):
        x, y = dataset[i]
