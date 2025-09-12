import numpy as np

from scipy.io import loadmat
from typing import Tuple, Any, List
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
    def __init__(self, add_temp: bool = False, add_alt: bool = False, **kwargs: Any):
        self.add_temp = add_temp
        self.add_alt = add_alt
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        data_all = loadmat(self.data_dir + "WildPPG.mat")
        arrays = []
        for participant in self.participants:
            # Load PPG signal and heart rate values
            ppg = data_all["data_ppg_ankle"][participant, 0]
            hr = data_all["data_bpm_values"][participant][0].astype(float)
            activity = data_all["data_imu_ankle"][participant][0]
            # temperature = data_all["data_temp_chest"][participant][0]
            # altitude = data_all["data_altitude_values"][participant][0]
            # temperature = data_all["data_imu_ankle"][participant][0]
            # altitude = data_all["data_imu_chest"][participant][0]

            # impute the values for hr and activity
            hr[hr < 30] = np.nan
            nans, x = nan_helper(hr)
            hr[nans] = np.interp(x(nans), x(~nans), hr[~nans])

            mask_activity = np.isnan(activity) | np.isinf(activity)
            activity[mask_activity] = np.nan
            nans, x = nan_helper(activity)
            activity[nans] = np.interp(x(nans), x(~nans), activity[~nans])

            mask_ppg = ~np.isnan(ppg).any(axis=1) & ~np.isinf(ppg).any(axis=1)
            ppg = ppg[mask_ppg]
            if self.use_heart_rate:
                series = hr  # (W, 1)
                if self.use_dynamic_features:
                    series = np.concatenate((series, activity), axis=1)  # shape (W, 2)
                    if self.add_temp:
                        series = np.concatenate((series, temperature), axis=1)  # (W,3)
                    if self.add_alt:
                        series = np.concatenate((series, altitude), axis=1)  # (W, 4)
            else:
                series = ppg[:, :, np.newaxis]  # shape (W, 200, 1)
                if self.use_dynamic_features:
                    repeated_activity = np.repeat(activity, repeats=200, axis=1)[
                        :, :, np.newaxis
                    ]  # shape (W, 200, 1)

                    series = np.concatenate(
                        (series, repeated_activity[mask_ppg]), axis=-1
                    )

            arrays.append(series)

        combined = np.concatenate(arrays, axis=0)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        min = np.min(combined, axis=0)
        max = np.max(combined, axis=0)

        return arrays, [mean, std, min, max]


class WildPPGDataModule(BaseDataModule):
    def __init__(
        self,
        use_heart_rate: bool = False,
        train_participants: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        val_participants: List[int] = [10, 11],
        test_participants: List[int] = [9, 12, 13, 14, 15],
        add_temp: bool = False,
        add_alt: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.add_temp = add_temp
        self.add_alt = add_alt

    def setup(self, stage: str):
        common_args = {
            "data_dir": self.data_dir,
            "use_dynamic_features": self.use_dynamic_features,
            "use_static_features": self.use_static_features,
            "use_heart_rate": self.use_heart_rate,
            "look_back_window": self.look_back_window,
            "prediction_window": self.prediction_window,
            "target_channel_dim": self.target_channel_dim,
            "test_local": self.test_local,
            "train_frac": self.train_frac,
            "val_frac": self.val_frac,
            "add_temp": self.add_temp,
            "add_alt": self.add_alt,
        }
        if stage == "fit":
            self.train_dataset = WildPPGDataset(
                participants=self.train_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
            self.val_dataset = WildPPGDataset(
                participants=self.val_participants,
                return_whole_series=self.return_whole_series,
                **common_args,
            )
        if stage == "test":
            self.test_dataset = WildPPGDataset(
                participants=self.test_participants,
                return_whole_series=False,
                **common_args,
            )
