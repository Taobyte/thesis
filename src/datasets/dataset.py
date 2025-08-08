import numpy as np
import torch

from torch import Tensor
from torch.utils.data import Dataset
from numpy.typing import NDArray
from typing import Tuple


class HRDataset(Dataset[Tuple[Tensor, Tensor]]):
    def __init__(
        self,
        data_dir: str,
        participants: list[int],
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        use_heart_rate: bool = False,
        look_back_window: int = 32,
        prediction_window: int = 10,
        target_channel_dim: int = 1,
        test_local: bool = False,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
    ):
        self.data_dir = data_dir
        self.use_heart_rate = use_heart_rate
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.participants = participants
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.target_channel_dim = target_channel_dim

        self.data, [self.mean, self.std, self.min, self.max] = self.__read_data__()
        if test_local:
            transformed_data: list[NDArray[np.float32]] = []
            for series in self.data:
                length = len(series)
                val_end = int(length * (train_frac + val_frac))
                transformed_data.append(series[val_end - self.look_back_window :, :])
            self.data = transformed_data

        self.lengths = [len(series) - self.window_length + 1 for series in self.data]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window_length)]
        look_back_window = torch.from_numpy(window[: self.look_back_window, :])
        prediction_window = torch.from_numpy(window[self.look_back_window :, :])

        return look_back_window.float(), prediction_window.float()

    def __read_data__(
        self,
    ) -> Tuple[list[NDArray[np.float32]], list[NDArray[np.float32]]]:
        raise NotImplementedError()
