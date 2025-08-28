import numpy as np
import pandas as pd

from typing import Tuple, Any, List
from numpy.typing import NDArray

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


class MITBIHDataset(HRDataset):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> Tuple[List[NDArray[np.float32]], List[NDArray[np.float32]]]:
        df = pd.read_csv(self.data_dir + "/heart_rate.csv")
        loaded_series = []
        for participant in self.participants:
            series = df[participant].dropna().astype(np.float32).values
            series = series[:, np.newaxis]
            loaded_series.append(series)

        combined = np.concatenate(loaded_series, axis=0)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        min = np.min(combined, axis=0)
        max = np.max(combined, axis=0)

        return loaded_series, [mean, std, min, max]


class MITBIHDatamodule(BaseDataModule):
    def __init__(
        self,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7],
        val_participants: list[int] = [8, 9],
        test_participants: list[int] = [10, 11, 12],
        use_heart_rate: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str):
        common_args = dict(
            data_dir=self.data_dir,
            look_back_window=self.look_back_window,
            prediction_window=self.prediction_window,
            use_heart_rate=self.use_heart_rate,
            use_dynamic_features=self.use_dynamic_features,
            target_channel_dim=self.target_channel_dim,
            test_local=self.test_local,
            train_frac=self.train_frac,
            val_frac=self.val_frac,
        )
        if stage == "fit":
            self.train_dataset = MITBIHDataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
            self.val_dataset = MITBIHDataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
        if stage == "test":
            self.test_dataset = MITBIHDataset(
                participants=self.test_participants,
                return_whole_series=False,
                **common_args,
            )
