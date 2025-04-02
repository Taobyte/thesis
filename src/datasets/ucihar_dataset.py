import pandas as pd
import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from gluonts.dataset.field_names import FieldName
from gluonts.dataset.common import ListDataset
from gluonts.transform import (
    Chain,
    AddObservedValuesIndicator,
    InstanceSplitter,
    InstanceSampler,
    ExpectedNumInstanceSampler,
    FieldName,
    AddAgeFeature,  # Good feature to use when absolute time is unknown
    # DO NOT USE AddTimeFeatures
)
from gluonts.torch.batchify import batchify


def create_dataloader(
    stage: str,
    datadir: str,
    start_time: str,
    freq: str,
    sampler: InstanceSampler,
    look_back_window: int,
    prediction_window: int,
) -> DataLoader:
    start_time = pd.Timestamp(start_time, freq=freq)

    subject_train = np.loadtxt(datadir + "subject_train.txt")
    x_train = np.loadtxt(datadir + "X_train.txt")
    y_train = np.loadtxt(datadir + "y_train.txt")

    combined = np.concatenate([subject_train, y_train, x_train], axis=1)
    df = pd.DataFrame(combined)
    df.columns = ["subject", "action"] + [
        "feature" + str(i) for i in range(x_train.shape[1])
    ]

    train_time_series = []

    for subject in df["subject"].unique():
        target = df[df["subject"] == subject].iloc[2:]
        action_feature = df[df["subject"] == subject]["action"]

        data_entry = {
            FieldName.TARGET: target,
            FieldName.START: start_time,
            FieldName.ITEM_ID: subject,
            FieldName.FEAT_DYNAMIC_CAT: action_feature,
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
        instance_sampler=sampler,
        past_length=look_back_window,
        future_length=prediction_window,
        time_series_fields=[
            FieldName.OBSERVED_VALUES,
            FieldName.FEAT_DYNAMIC_CAT,
        ],
        dummy_value=0.0,
    )
    transformation = [instance_splitter]

    transformed_train_ds = train_ds.transform(transformation, is_train=True)

    train_dataloader = DataLoader(
        transformed_train_ds,  # Use the eagerly transformed dataset/iterable
        batch_size=32,  # Number of INSTANCES per batch
        shuffle=True,  # Shuffle the generated instances for training
        num_workers=0,  # Adjust as needed
        collate_fn=batchify,  # Use GluonTS batchify or default PyTorch collate
    )

    return train_dataloader


class UCIHARDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        freq: str = "20L",
        start_time: str = "2000-01-01 00:00:00",
        sampler: InstanceSampler = None,
    ):
        pass

    def __getitem__(self, idx: int) -> torch.Tensor:
        pass


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
