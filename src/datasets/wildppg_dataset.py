import numpy as np
import pandas as pd
import torch
import lightning as L

from typing import Tuple
from scipy.io import loadmat
from torch.utils.data import Dataset, DataLoader


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


def wildppg_load_data(
    datadir: str, participants: list[str], use_heart_rate: bool, use_activity_info: bool
) -> Tuple[list[np.ndarray], float, float]:
    data_all = loadmat(datadir + "WildPPG.mat")

    arrays = []
    for participant in participants:
        # Load PPG signal and heart rate values
        ppg = data_all["data_ppg_wrist"][participant, 0]
        hr = data_all["data_bpm_values"][participant][0].astype(float)
        activity = data_all["data_imu_wrist"][participant][0]

        # impute the values for hr and activity
        hr[hr < 30] = np.nan
        nans, x = nan_helper(hr)
        hr[nans] = np.interp(x(nans), x(~nans), hr[~nans])

        mask_activity = np.isnan(activity) | np.isinf(activity)
        activity[mask_activity] = np.nan
        nans, x = nan_helper(activity)
        activity[nans] = np.interp(x(nans), x(~nans), hr[~nans])

        # impute the activity values

        mask_ppg = ~np.isnan(ppg).any(axis=1) & ~np.isinf(ppg).any(axis=1)
        ppg = ppg[mask_ppg]
        if use_heart_rate:
            series = hr  # (W, 1)
            if use_activity_info:
                series = np.concatenate((series, activity), axis=1)  # shape (W, 2)
        else:
            series = ppg[:, :, np.newaxis]  # shape (W, 200, 1)
            if use_activity_info:
                repeated_activity = np.repeat(activity, repeats=200, axis=1)[
                    :, :, np.newaxis
                ]  # shape (W, 200, 1)

                series = np.concatenate((series, repeated_activity[mask_ppg]), axis=-1)

        arrays.append(series)

    combined_series = np.concatenate(
        [arr.reshape(-1, arr.shape[-1]) for arr in arrays], axis=0
    )
    global_mean = np.mean(combined_series, axis=0, keepdims=True)[np.newaxis, :, :]
    global_std = np.std(combined_series, axis=0, keepdims=True)[np.newaxis, :, :]

    return arrays, global_mean, global_std


class WildPPGDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        use_heart_rate: bool = False,
        look_back_window: int = 320,
        prediction_window: int = 128,
        participants: list[str] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
        use_activity_info: bool = False,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        self.arrays, self.global_mean, self.global_std = wildppg_load_data(
            datadir, participants, use_heart_rate, use_activity_info
        )

        self.base_channel_dim = 1
        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info
        assert self.window <= 200  # window lengths of WildPPG is 200
        if use_heart_rate:
            self.lengths = [(len(arr) - self.window + 1) for arr in self.arrays]
        else:
            self.lengths = [len(arr) * (200 - self.window + 1) for arr in self.arrays]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        if self.use_heart_rate:
            index = idx - self.cumulative_lengths[file_idx]
            window = self.arrays[file_idx][index : (index + self.window)]
        else:
            participant_pos = idx - self.cumulative_lengths[file_idx]
            row_index = participant_pos // (200 - self.window + 1)
            window_pos = participant_pos % (200 - self.window + 1)
            window = self.arrays[file_idx][row_index][
                window_pos : window_pos + self.window
            ]

        look_back_window = torch.from_numpy(window[: self.look_back_window])
        prediction_window = torch.from_numpy(
            window[self.look_back_window :, : self.base_channel_dim]
        )

        return look_back_window.float(), prediction_window.float()


class WildPPGDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list[str] = [
            "an0",
            "e61",
            "fex",
            "k2s",
            "kjd",
            "l38",
            "n31",
            "ngh",
            "p5d",
            "p9p",
        ],
        val_participants: list[str] = ["qm9", "ssx", "trh"],
        test_participants: list[str] = ["tz8", "u7y", "w4p"],
        freq: int = 25,
        name: str = "wildppg",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.freq = freq
        self.name = name

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = WildPPGDataset(
                self.data_dir,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_activity_info,
            )
            self.val_dataset = WildPPGDataset(
                self.data_dir,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = WildPPGDataset(
                self.data_dir,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_activity_info,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    from tqdm import tqdm

    datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/wildppg_preprocessed/"
    dataset = WildPPGDataset(datadir, use_activity_info=True)
    for i in tqdm(range(len(dataset))):
        look_back_window, prediction_window = dataset[i]
        print(look_back_window.shape)
        break
