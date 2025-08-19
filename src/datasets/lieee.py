import numpy as np
import torch

from torch import Tensor
from typing import Any, Tuple
from numpy.typing import NDArray

from src.datasets.ieee_dataset import IEEEDataModule
from src.datasets.ieee_dataset import IEEEDataset


class LIEEEDataset(IEEEDataset):
    def __init__(
        self,
        flag: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        normalization: str = "none",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert len(self.participants) == 1
        self.flag = flag
        self.normalization = normalization
        timeseries = self.data[0]
        train_end = int(len(timeseries) * train_frac)
        val_end = int(len(timeseries) * (train_frac + val_frac))
        if flag == "train":
            timeseries = timeseries[:train_end]
        elif flag == "val":
            timeseries = timeseries[train_end - self.look_back_window : val_end]
        elif flag == "test":
            timeseries = timeseries[val_end - self.look_back_window :]
        else:
            raise NotImplementedError()
        self.timeseries = timeseries
        self.mean = np.mean(timeseries, axis=0)
        self.std = np.std(timeseries, axis=0)
        self.min = np.min(timeseries, axis=0)
        self.max = np.max(timeseries, axis=0)

    def __len__(self) -> int:
        return len(self.timeseries) - self.window_length + 1

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        window_pos = idx
        look_back_window = self.timeseries[
            window_pos : window_pos + self.look_back_window
        ]
        prediction_window = self.timeseries[
            window_pos + self.look_back_window : window_pos + self.window_length
        ]

        look_back_window = torch.from_numpy(look_back_window).float()
        prediction_window = torch.from_numpy(prediction_window[:, :]).float()

        return look_back_window, prediction_window

    def get_normalized_timeseries(self) -> NDArray[np.float32]:
        assert self.flag == "train"
        if self.normalization == "global":
            return (self.timeseries - self.mean) / (self.std + 1e-8)
        elif self.normalization == "minmax":
            return (self.timeseries - self.min) / (self.max - self.min)
        elif self.normalization == "difference":
            ts = self.timeseries
            if ts.ndim == 1:
                pad = np.zeros((1,))
            else:
                pad = np.zeros((1, ts.shape[-1]))
            return np.concatenate([pad, np.diff(ts, axis=0)], axis=0)
        return self.timeseries


class LIEEE(IEEEDataModule):
    def __init__(
        self,
        participant: int = 1,
        use_heart_rate: bool = True,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate
        self.train_frac = train_frac
        self.val_frac = val_frac
        self.participant = participant

    def setup(self, stage: str):
        common_args = dict(
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            data_dir=self.data_dir,
            participants=[self.participant],
            use_heart_rate=self.use_heart_rate,
            target_channel_dim=self.target_channel_dim,
            look_back_window=self.look_back_window,
            prediction_window=self.prediction_window,
            normalization=self.normalization,
            use_dynamic_features=self.use_dynamic_features,
        )
        if stage == "fit":
            self.train_dataset = LIEEEDataset(flag="train", **common_args)
            self.val_dataset = LIEEEDataset(flag="val", **common_args)
        if stage == "test":
            self.test_dataset = LIEEEDataset(flag="test", **common_args)
