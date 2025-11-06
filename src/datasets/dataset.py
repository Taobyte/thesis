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
        look_back_window: int = 30,
        prediction_window: int = 10,
        target_channel_dim: int = 1,
        test_local: bool = False,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        return_whole_series: bool = False,
        is_test_dataset: bool = False,
        max_eval_look_back_window: int = 60,
        downsample_factor: int = 1,
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

        downsampled_data: list[NDArray[np.float32]] = []
        for series in self.data:
            T = len(series)
            usable_len = (T // downsample_factor) * downsample_factor
            trimmed = series[:usable_len, :]
            reshaped = trimmed.reshape(-1, downsample_factor, series.shape[1])
            downsampled = np.nanmean(reshaped, axis=1)
            downsampled_data.append(downsampled.astype(np.float32))
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
        elif is_test_dataset:
            # for the ablation study, we want to evaluate on the exact same prediction windows
            # the problem is that the # of windows changes depending on the lookback window length
            # thus we have to remove the beginning of the series if the lbw < max_lbw
            test_start_idx = max_eval_look_back_window - look_back_window
            transformed_data: list[NDArray[np.float32]] = []
            for series in self.data:
                transformed_data.append(series[test_start_idx:, :])
            self.data = transformed_data

        self.lengths = [(len(v) - self.window_length + 1) for v in self.data]
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
            start = idx - self.cumulative_lengths[file_idx]
            window = self.data[file_idx][start : start + self.window_length]
            look_back_window = torch.from_numpy(window[: self.look_back_window, :])
            prediction_window = torch.from_numpy(window[self.look_back_window :, :])
            return look_back_window.float(), prediction_window.float()

    def __read_data__(
        self,
    ) -> list[NDArray[np.float32]]:
        raise NotImplementedError()
