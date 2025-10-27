import numpy as np

from scipy.interpolate import PchipInterpolator
from typing import Any, List
from numpy.typing import NDArray

from src.datasets.utils import BaseDataModule
from src.datasets.dataset import HRDataset


def spline_fill_all_nan(
    y: NDArray[np.float32], kind: str = "pchip"
) -> NDArray[np.float32]:
    """
    Interpolate all NaN values in a 1D HR series using a smooth spline.
    kind: 'pchip' (default, safe for HR), 'akima', or 'cubic'
    """
    y = np.asarray(y, dtype=np.float32).reshape(-1)
    x = np.arange(len(y), dtype=np.float32)
    mask = np.isfinite(y)

    if mask.sum() < 2:
        return y  # not enough points to interpolate

    if kind == "pchip":
        interp = PchipInterpolator(x[mask], y[mask], extrapolate=True)
    elif kind == "akima":
        from scipy.interpolate import Akima1DInterpolator

        interp = Akima1DInterpolator(x[mask], y[mask])
    elif kind == "cubic":
        from scipy.interpolate import CubicSpline

        interp = CubicSpline(x[mask], y[mask], bc_type="natural", extrapolate=True)
    else:
        raise ValueError("kind must be 'pchip', 'akima', or 'cubic'")

    y_filled = interp(x)
    return y_filled.astype(np.float32)


class WildPPGDataset(HRDataset):
    def __init__(
        self,
        sensor_location: str = "ankle",
        imu_features: list[str] = ["rms", "mean"],
        **kwargs: Any,
    ):
        self.sensor_location = sensor_location
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
            hr = hr_arr[participant].astype(float)
            print(f"Participant {participant} contains {sum(hr < 30)}")
            hr[hr < 30] = np.nan

            # interpolate nan values with smoothing spline
            hr_filled = spline_fill_all_nan(hr, kind="pchip")
            series = hr_filled[:, None]  # (W, 1)

            # IMU features
            if self.use_dynamic_features:
                imu_participant = imus_arr[participant]
                features: list[NDArray[np.float32]] = []
                for feature in self.imu_features:
                    features.append(
                        imu_participant[self.sensor_location + "_" + feature]
                    )

                assert len(features) > 0, (
                    "ATTENTION: you set use_dynamic_features=True, but pass no imu_features"
                )
                series = np.concatenate([series] + features, axis=1)
            assert not np.isnan(series).any(), (
                f"series contains nan values at {np.where(np.isnan(series))}"
            )
            arrays.append(series)

        return arrays


class WildPPGDataModule(BaseDataModule):
    def __init__(
        self,
        train_participants: List[int] = [0, 1, 2, 3, 4, 5, 6, 7, 8],
        val_participants: List[int] = [10, 11],
        test_participants: List[int] = [9, 12, 13, 14, 15],
        sensor_location: str = "ankle",
        imu_features: list[str] = ["rms"],
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.sensor_location = sensor_location
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
            "sensor_location": self.sensor_location,
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
