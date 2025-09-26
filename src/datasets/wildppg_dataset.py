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
    def __init__(
        self,
        add_rmse: bool,
        add_enmo: bool,
        add_mad: bool,
        add_perc: bool,
        add_jerk: bool,
        add_cadence: bool,
        add_rmse_last2: bool,
        shift: bool,
        add_alt: bool,
        add_temp: bool,
        **kwargs: Any,
    ):
        self.shift = shift

        self.add_rmse = add_rmse
        self.add_enmo = add_enmo
        self.add_mad = add_mad
        self.add_perc = add_perc
        self.add_jerk = add_jerk
        self.add_cadence = add_cadence
        self.add_rmse_last2 = add_rmse_last2

        self.add_alt = add_alt
        self.add_temp = add_temp
        super().__init__(**kwargs)

    def __read_data__(
        self,
    ) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        data_all = loadmat(self.data_dir + "WildPPG.mat")
        arrays: list[NDArray[np.float32]] = []
        for participant in self.participants:
            hr = data_all["data_bpm_values"][participant][0].astype(float)
            activity = data_all["data_imu_ankle"][participant][0]
            rmse = data_all["rmse"][participant][0]
            enmo = data_all["enmo"][participant][0]
            mad = data_all["mad"][participant][0]
            perc = data_all["perc"][participant][0]
            jerk = data_all["jerk"][participant][0]
            cadence = data_all["cadence"][participant][0]
            rmse_last2 = data_all["rmse_last2"][participant][0]
            # ankle = data_all["data_imu_ankle"][participant][0]
            # wrist = data_all["data_imu_wrist"][participant][0]
            # chest = data_all["data_imu_chest"][participant][0]
            # activity = (1 / 3) * (ankle + wrist + chest)

            # if (altitude < 100).any():
            #     print("altitude < 100")

            # impute the values for hr and activity
            hr[hr < 30] = np.nan
            nans, x = nan_helper(hr)
            hr[nans] = np.interp(x(nans), x(~nans), hr[~nans])

            mask_activity = np.isnan(activity) | np.isinf(activity)
            activity[mask_activity] = np.nan
            nans, x = nan_helper(activity)
            activity[nans] = np.interp(x(nans), x(~nans), activity[~nans])

            if self.add_temp:
                temperature = data_all["data_temp_chest"][participant][0]
                temperature[temperature < 30] = np.nan
                nans, x = nan_helper(temperature)
                temperature[nans] = np.interp(x(nans), x(~nans), temperature[~nans])

            if self.add_alt:
                altitude = data_all["data_altitude_values"][participant][0]
                altitude[altitude < 100] = np.nan
                nans, x = nan_helper(altitude)
                altitude[nans] = np.interp(x(nans), x(~nans), altitude[~nans])

            if self.use_heart_rate:
                series = hr  # (W, 1)
                # IMU features
                if self.use_dynamic_features:
                    series = np.concatenate((series, activity), axis=1)  # shape (W, 2)
                if self.add_rmse:
                    series = np.concatenate((series, rmse), axis=1)  # (W, 4)
                if self.add_enmo:
                    series = np.concatenate((series, enmo), axis=1)  # (W, 4)
                if self.add_mad:
                    series = np.concatenate((series, mad), axis=1)  # (W, 4)
                if self.add_perc:
                    series = np.concatenate((series, perc), axis=1)  # (W, 4)
                if self.add_jerk:
                    series = np.concatenate((series, jerk), axis=1)  # (W, 4)
                if self.add_rmse_last2:
                    series = np.concatenate((series, rmse_last2), axis=1)  # (W, 4)
                if self.add_cadence:
                    series = np.concatenate((series, cadence), axis=1)  # (W, 4)

                # other exogenous covariates
                if self.add_temp:
                    series = np.concatenate((series, temperature), axis=1)  # (W,3)
                if self.add_alt:
                    series = np.concatenate((series, altitude), axis=1)  # (W, 4)

            arrays.append(series)

        if self.shift and self.use_dynamic_features:
            processed_arrays: list[NDArray[np.float32]] = []
            for array in arrays:
                hr_shifted = array[:-1, : self.target_channel_dim]
                imu_shifted = array[1:, self.target_channel_dim :]
                processed_arrays.append(
                    np.concatenate([hr_shifted, imu_shifted], axis=1)
                )

            arrays = processed_arrays

        combined = np.concatenate(arrays, axis=0)
        mean = np.mean(combined, axis=0)
        std = np.std(combined, axis=0)
        min = np.min(combined, axis=0)
        max = np.max(combined, axis=0)

        return arrays, [mean, std, min, max]


class WildPPGDataModule(BaseDataModule):
    def __init__(
        self,
        use_heart_rate: bool = True,
        train_participants: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        val_participants: List[int] = [10, 11],
        test_participants: List[int] = [9, 12, 13, 14, 15],
        add_rmse: bool = False,
        add_enmo: bool = False,
        add_mad: bool = False,
        add_perc: bool = False,
        add_jerk: bool = False,
        add_cadence: bool = False,
        add_rmse_last2: bool = False,
        shift: bool = False,
        add_alt: bool = False,
        add_temp: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.use_heart_rate = use_heart_rate

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.add_rmse = add_rmse
        self.add_enmo = add_enmo
        self.add_mad = add_mad
        self.add_perc = add_perc
        self.add_jerk = add_jerk
        self.add_cadence = add_cadence
        self.add_rmse_last2 = add_rmse_last2

        self.shift = shift
        self.add_alt = add_alt
        self.add_temp = add_temp

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
            "add_rmse": self.add_rmse,
            "add_enmo": self.add_enmo,
            "add_mad": self.add_mad,
            "add_perc": self.add_perc,
            "add_jerk": self.add_jerk,
            "add_cadence": self.add_cadence,
            "add_rmse_last2": self.add_rmse_last2,
            "shift": self.shift,
            "add_alt": self.add_alt,
            "add_temp": self.add_temp,
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
