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
            y60 = x[:n].reshape(-1, factor, C).mean(axis=1)
            downsampled_data.append(y60)
        self.data = downsampled_data

        combined_series = np.concatenate(self.data, axis=0)
        self.mean = np.mean(combined_series, axis=0)
        self.std = np.std(combined_series, axis=0)
        self.min = np.min(combined_series, axis=0)
        self.max = np.max(combined_series, axis=0)
        hr = combined_series[:, 0]
        self.hr_diff_quantile = np.array(
            np.quantile(
                np.abs(np.diff(hr, prepend=hr[0])),
                0.9,
            )
        )
        # print(f"HR Diff 0.9 Quantile: {self.hr_diff_quantile:.4f}")

        if test_local:
            transformed_data: list[NDArray[np.float32]] = []
            for series in self.data:
                length = len(series)
                val_end = int(length * (train_frac + val_frac))
                transformed_data.append(series[val_end - self.look_back_window :, :])
            self.data = transformed_data

        self.lengths = [len(series) - self.window_length + 1 for series in self.data]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = (
            self.cumulative_lengths[-1] if not return_whole_series else len(self.data)
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
            index = idx - self.cumulative_lengths[file_idx]
            window = self.data[file_idx][index : (index + self.window_length)]
            look_back_window = torch.from_numpy(window[: self.look_back_window, :])
            prediction_window = torch.from_numpy(window[self.look_back_window :, :])

            return look_back_window.float(), prediction_window.float()

    def __read_data__(
        self,
    ) -> list[NDArray[np.float32]]:
        raise NotImplementedError()
