import numpy as np

from typing import Any, List
from numpy.typing import NDArray

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


class IEEEDataset(HRDataset):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> List[NDArray[np.float32]]:
        loaded_series = []

        for participant in self.participants:
            data = np.load(self.data_dir + f"IEEE_{participant}.npz")
            ppg = data["ppg"]  # shape (W, 200, 1)
            acc = data["acc"]  # shape (W, 200, 1)
            bpm = data["bpms"]  # shape (W, 1)

            if self.use_heart_rate and self.use_dynamic_features:
                imu_mean = np.mean(acc, axis=1, keepdims=True).squeeze(
                    -1
                )  # shape (W, 1)
                imu_var = np.var(acc, axis=1, keepdims=True).squeeze(-1)
                imu_power = np.mean(acc**2, axis=1, keepdims=True).squeeze(-1)
                imu_energy = np.sum(acc**2, axis=1, keepdims=True).squeeze(-1)
                series = np.concatenate((bpm, imu_mean), axis=1)
            elif self.use_heart_rate and not self.use_dynamic_features:
                series = bpm
            elif not self.use_heart_rate and self.use_dynamic_features:
                series = np.concatenate((ppg, acc), axis=2)
            else:
                series = ppg

            loaded_series.append(series)

        #  np.savez(
        #      self.data_dir + f"IEEE_{participant}_R.npz",
        #      imu=imu_mean[:, 0],
        #      bpm=bpm[:, 0],
        #  )

        return loaded_series


class IEEEDataModule(BaseDataModule):
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
            self.train_dataset = IEEEDataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
            self.val_dataset = IEEEDataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                participants=self.test_participants,
                return_whole_series=False,
                **common_args,
            )
