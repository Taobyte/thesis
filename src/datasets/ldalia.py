import numpy as np
import torch

from typing import Tuple, Any
from torch import Tensor

from src.datasets.dalia_dataset import DaLiADataset, DaLiADataModule


class LDaliaDataset(DaLiADataset):
    def __init__(
        self,
        flag: str = "train",
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert (
            self.drop_windows_with_invalid_hr is False
        )  # ensures that we interpolate the data for local forecasting
        assert len(self.participants) == 1
        self.flag = flag
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
        return (
            1
            if self.return_whole_series
            else len(self.timeseries) - self.window_length + 1
        )

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        if self.return_whole_series:
            tensor_series = torch.from_numpy(self.timeseries).float()
            return tensor_series
        else:
            start = idx
            window = self.timeseries[start : (start + self.window_length)]
            look_back_window = torch.from_numpy(window[: self.look_back_window, :])
            prediction_window = torch.from_numpy(window[self.look_back_window :, :])

            return look_back_window.float(), prediction_window.float()


class LDalia(DaLiADataModule):
    def __init__(
        self,
        participant: int = 1,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.participant = participant
        self.train_frac = train_frac
        self.val_frac = val_frac

    def setup(self, stage: str = "fit"):
        common_args: dict[str, Any] = dict(
            train_frac=self.train_frac,
            val_frac=self.val_frac,
            data_dir=self.data_dir,
            participants=[self.participant],
            use_dynamic_features=self.use_dynamic_features,
            use_static_features=self.use_static_features,
            look_back_window=self.look_back_window,
            prediction_window=self.prediction_window,
            target_channel_dim=self.target_channel_dim,
        )
        if stage == "fit":
            self.train_dataset = LDaliaDataset(
                flag="train",
                return_whole_series=self.return_whole_series,
                **common_args,
            )
            self.val_dataset = LDaliaDataset(
                flag="val", return_whole_series=self.return_whole_series, **common_args
            )
        if stage == "test":
            self.test_dataset = LDaliaDataset(
                flag="test", return_whole_series=False, **common_args
            )
