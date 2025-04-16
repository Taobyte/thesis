import numpy as np
import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset
from scipy.io import loadmat


def ieee_load_data(
    datadir: str, participants: list[int], use_heart_rate: bool, use_activity_info: bool
):
    """
    Load time series data from the IEEE_Big.mat dataset for selected participants.

    Parameters
    ----------
    datadir : str
        Path to the directory containing the "IEEE_Big.mat" file.
    participants : list[int]
        List of participant IDs (1-based indices) to load data for.
    use_heart_rate : bool
        If True, loads heart rate time series directly for each participant.
        If False, loads multichannel data (e.g., multiple signals per participant).

    Returns
    -------
    list
        - If `use_heart_rate` is True:
            A list of NumPy arrays, one per participant, each of shape (T,), where T is the time series length.
        - If `use_heart_rate` is False:
            A list of lists of NumPy arrays. Each inner list corresponds to one participant and contains multiple time series arrays
            (each of shape (T,)) representing different channels/signals.
    """

    data = loadmat(datadir + "IEEE_Big.mat")["whole_dataset"]
    if use_heart_rate:
        return [data[i - 1][1].squeeze(-1) for i in participants]
    else:
        loaded_series = []
        for i in participants:
            arr = data[i - 1][0]
            participant_series = []
            for j in range(len(arr)):
                participant_series.append(arr[j, :])  # (200,)
            loaded_series.append(participant_series)
        return loaded_series


class IEEEDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        look_back_window: int,
        prediction_window: int,
        participants: list[int],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
    ):
        self.datadir = datadir
        self.look_back_window = look_back_window
        self.predicition_window = prediction_window
        self.window_length = look_back_window + prediction_window
        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info

        self.data = ieee_load_data(datadir, participants, use_heart_rate)

        assert self.window_length <= 200, (
            f"window_length: {self.window_length}: IEEE for PPG contains only time series of length 200!"
        )

        self.series_per_participant = [len(series) for series in self.data]
        self.n_windows_per_participant = [
            n_series * (200 - self.window_length + 1)
            for n_series in self.series_per_participant
        ]

        if use_heart_rate:
            self.cumulative_sum = np.cumsum(
                [0]
                + [
                    (l - self.window_length + 1)
                    for l in self.series_per_participant
                    if l >= self.window_length
                ]
            )
            self.data = [d for d in self.data if len(d) >= self.window_length]

        else:
            self.cumulative_sum = np.cumsum([0] + self.n_windows_per_participant)
        self.total_length = self.cumulative_sum[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        participant_idx = np.searchsorted(self.cumulative_sum, idx, side="right") - 1
        index = idx - self.cumulative_sum[participant_idx]

        if self.use_heart_rate:
            data = self.data[participant_idx]
            window_pos = index
        else:
            serie_idx = index // (200 - self.window_length + 1)
            window_pos = index % (200 - self.window_length + 1)
            data = self.data[participant_idx][serie_idx]

        look_back_window = data[window_pos : window_pos + self.look_back_window]
        prediction_window = data[
            window_pos + self.look_back_window : window_pos + self.window_length
        ]

        look_back_window = torch.from_numpy(look_back_window[:, np.newaxis]).float()
        prediction_window = torch.from_numpy(prediction_window[:, np.newaxis]).float()

        return look_back_window, prediction_window


class IEEEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        num_workers: int = 8,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list[int] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        val_participants: list[int] = [15, 16, 17, 18],
        test_participants: list[int] = [19, 20, 21, 22],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )
            self.val_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
                self.use_heart_rate,
                self.use_activity_info,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers
        )


if __name__ == "__main__":
    datadir = "C:/Users/cleme/ETH/Master/Thesis/data/euler/IEEEPPG/"
    data = ieee_load_data(datadir, [1, 2, 3], False)
    print(data[0][0].shape)
