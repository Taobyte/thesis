import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import Tuple, Union


class HRDataset(Dataset[Union[Tensor, Tuple[Tensor, Tensor]]]):
    def __init__(
        self,
        data_dir: str,
        participants: list[int],
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        look_back_window: int = 32,
        prediction_window: int = 10,
        target_channel_dim: int = 1,
        test_local: bool = False,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        return_whole_series: bool = False,
    ):
        self.data_dir = data_dir
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.participants = participants
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.target_channel_dim = target_channel_dim
        self.return_whole_series = return_whole_series

        self.data = self.__read_data__()
        factor = 1
        downsampled_data: list[NDArray[np.float32]] = []
        for x in self.data:
            T, C = x.shape  # C should be 2
            n = (T // factor) * factor
            y60 = np.nanmean(x[:n].reshape(-1, factor, C), axis=1)
            downsampled_data.append(y60)
        self.data = downsampled_data

        combined_series = np.concatenate(self.data, axis=0)
        self.mean = np.nanmean(combined_series, axis=0)
        self.std = np.nanstd(combined_series, axis=0)
        self.min = np.nanmin(combined_series, axis=0)
        self.max = np.nanmax(combined_series, axis=0)
        hr = combined_series[:, 0]

        valid = np.isfinite(hr)
        raw_diffs = np.diff(hr)
        keep = valid[1:] & valid[:-1]
        diffs_consecutive_valid = np.abs(raw_diffs[keep])
        self.hr_diff_quantile = float(np.quantile(diffs_consecutive_valid, 0.9))
        # print(f"HR Diff 0.9 Quantile: {self.hr_diff_quantile:.4f}")

        if test_local:
            transformed_data: list[NDArray[np.float32]] = []
            for series in self.data:
                length = len(series)
                val_end = int(length * (train_frac + val_frac))
                transformed_data.append(series[val_end - self.look_back_window :, :])
            self.data = transformed_data

        # drop invalid windows (i.e hr < 30 or hr == nan or hr == inf)
        self.drop_windows_with_invalid_hr = True
        self.min_hr = 30
        self.valid_starts_per_series: list[NDArray[np.float32]] = []
        if self.drop_windows_with_invalid_hr and not self.return_whole_series:
            K = self.window_length
            ones = np.ones(K, dtype=np.int32)

            for series in self.data:
                hr_series = series[:, 0]
                invalid = (~np.isfinite(hr_series)) | (hr_series < self.min_hr)
                if len(series) < K:
                    self.valid_starts_per_series.append(np.array([], dtype=np.int64))
                    continue
                roll_invalid = np.convolve(invalid.astype(np.int32), ones, mode="valid")
                valid_starts = np.nonzero(roll_invalid == 0)[0]
                self.valid_starts_per_series.append(valid_starts.astype(np.int64))
        else:
            # default: all contiguous windows are allowed
            interpolated_data: list[NDArray[np.float32]] = []
            for series in self.data:
                series_interp = series.copy()
                y = series_interp[:, 0]
                if np.any(np.isnan(y)):
                    # we assume that only the hr time series contains NANS (only wildppg contains nan values)
                    assert not np.any(np.isnan(series_interp[:, 1:]))
                    nans = np.isnan(y)
                    not_nans = ~nans
                    series_interp[nans, 0] = np.interp(
                        np.flatnonzero(nans), np.flatnonzero(not_nans), y[not_nans]
                    )
                interpolated_data.append(series_interp)
                n = max(0, len(series) - self.window_length + 1)
                self.valid_starts_per_series.append(np.arange(n, dtype=np.int64))

            self.data = interpolated_data

        # lengths/cumulative/total based on valid starts
        self.lengths = [len(v) for v in self.valid_starts_per_series]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = (
            self.cumulative_lengths[-1]
            if not self.return_whole_series
            else len(self.data)
        )

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        if self.return_whole_series:
            series = self.data[idx]
            tensor_series = torch.from_numpy(series).float()
            return tensor_series
        else:
            file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
            pos_in_series = idx - self.cumulative_lengths[file_idx]

            start = self.valid_starts_per_series[file_idx][pos_in_series]
            window = self.data[file_idx][start : start + self.window_length]

            look_back_window = torch.from_numpy(window[: self.look_back_window, :])
            prediction_window = torch.from_numpy(window[self.look_back_window :, :])

            return look_back_window.float(), prediction_window.float()

    def __read_data__(
        self,
    ) -> list[NDArray[np.float32]]:
        raise NotImplementedError()
