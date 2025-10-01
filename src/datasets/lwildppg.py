import numpy as np
import torch

from typing import Any, Union, Tuple

from src.datasets.wildppg_dataset import WildPPGDataset, WildPPGDataModule


class LWildPPGDataset(WildPPGDataset):
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
        return (
            1
            if self.return_whole_series
            else len(self.timeseries) - self.window_length + 1
        )

    def __getitem__(
        self, idx: int
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        if self.return_whole_series:
            tensor_series = torch.from_numpy(self.timeseries).float()
            return tensor_series
        else:
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


class LWildPPG(WildPPGDataModule):
    def __init__(
        self,
        participant: int = 1,
        use_heart_rate: bool = False,
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
        if stage == "fit":
            self.train_dataset = LWildPPGDataset(
                flag="train",
                return_whole_series=self.return_whole_series,
                participants=[self.participant],
                **self.common_args,
            )
            self.val_dataset = LWildPPGDataset(
                flag="val",
                return_whole_series=self.return_whole_series,
                participants=[self.participant],
                **self.common_args,
            )
        if stage == "test":
            self.test_dataset = LWildPPGDataset(
                flag="test",
                return_whole_series=False,
                participants=[self.participant],
                **self.common_args,
            )
