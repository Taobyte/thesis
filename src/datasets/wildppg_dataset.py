import numpy as np

from typing import Any, List
from numpy.typing import NDArray

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


class WildPPGDataset(HRDataset):
    def __init__(
        self,
        imu_features: list[str] = ["ankle_rms", "ankle_mean"],
        **kwargs: Any,
    ):
        self.imu_features = imu_features
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> list[NDArray[np.float32]]:
        with np.load(self.data_dir + "WildPPG.npz", allow_pickle=True) as data:
            hr_arr = data["hr"]
            imus_arr = data["imus"]
        arrays: list[NDArray[np.float32]] = []
        for participant in self.participants:
            hr = hr_arr[participant][:, np.newaxis].astype(float)

            hr[hr < 30] = np.nan
            nans, x = nan_helper(hr)
            hr[nans] = np.interp(x(nans), x(~nans), hr[~nans])

            series = hr  # (W, 1)
            # IMU features
            if self.use_dynamic_features:
                imu_participant = imus_arr[participant]
                features: list[NDArray[np.float32]] = []
                for feature in self.imu_features:
                    features.append(imu_participant[feature][: len(hr), np.newaxis])

                assert len(features) > 0, (
                    "ATTENTION: you set use_dynamic_features=True, but pass no imu_features"
                )
                series = np.concatenate([series] + features, axis=1)
            arrays.append(series)

        return arrays


class WildPPGDataModule(BaseDataModule):
    def __init__(
        self,
        train_participants: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        val_participants: List[int] = [10, 11],
        test_participants: List[int] = [9, 12, 13, 14, 15],
        imu_features: list[str] = ["ankle_rms"],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.imu_features = imu_features

        self.common_args: dict[str, Any] = {
            "data_dir": self.data_dir,
            "use_dynamic_features": self.use_dynamic_features,
            "use_static_features": self.use_static_features,
            "look_back_window": self.look_back_window,
            "prediction_window": self.prediction_window,
            "target_channel_dim": self.target_channel_dim,
            "test_local": self.test_local,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "imu_features": self.imu_features,
        }

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = WildPPGDataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
            self.val_dataset = WildPPGDataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **self.common_args,
            )
        if stage == "test":
            self.test_dataset = WildPPGDataset(
                participants=self.test_participants,
                return_whole_series=False,
                **self.common_args,
            )
