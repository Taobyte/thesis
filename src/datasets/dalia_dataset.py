import numpy as np
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple
from scipy import signal


class DaLiADataset(Dataset):
    def __init__(
        self,
        path: Path,
        participants: list[int],
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
        look_back_window: int = 32,
        prediction_window: int = 10,
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        self.participants = participants

        self.data = []
        self.lengths = []
        for i in self.participants:
            label = "S" + str(i)
            data_path = Path(path) / (label + ".npz")
            data = np.load(data_path)
            series = data["ecg"] if use_heart_rate else data["bvp"]
            assert len(series) >= self.window
            if use_activity_info:
                activity = data["wrist_acc"]
                activity = np.sqrt(
                    activity[:, 0] ** 2 + activity[:, 1] ** 2, activity[:, 2] ** 2
                )[:, np.newaxis]
                activity_resampled = signal.resample(activity, len(series))
                series = np.concatenate((series, activity_resampled), axis=1)

            self.data.append(series)
            self.lengths.append(len(series))

        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.data[file_idx][index : (index + self.window)]
        look_back_window = torch.from_numpy(
            window[: self.look_back_window, 0]
        ).unsqueeze(-1)
        prediction_window = torch.from_numpy(
            window[self.look_back_window :, 0]
        ).unsqueeze(-1)

        return look_back_window.float(), prediction_window.float()


class DaLiADataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        use_heart_rate: bool = False,
        use_activity_info: bool = False,
        look_back_window: int = 128,
        prediction_window: int = 64,
        train_participants: list = [2, 3, 4, 5, 6, 7, 8, 12, 15],
        val_participants: list = [9, 10, 11],
        test_participants: list = [1, 13, 14],
    ):
        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.use_heart_rate = use_heart_rate
        self.use_activity_info = use_activity_info

        self.train_participants = train_participants
        self.val_participants = val_participants
        self.test_participants = test_participants

    def setup(self, stage: str = None):
        if stage == "fit":
            self.train_dataset = DaLiADataset(
                self.data_dir,
                self.train_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
            )
            self.val_dataset = DaLiADataset(
                self.data_dir,
                self.val_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
            )
        if stage == "test":
            self.test_dataset = DaLiADataset(
                self.data_dir,
                self.test_participants,
                self.use_heart_rate,
                self.use_activity_info,
                self.look_back_window,
                self.prediction_window,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    from tqdm import tqdm

    modes = ["train", "val", "test"]
    use_heart_rate = [True, False]
    path = Path("C:/Users/cleme/ETH/Master/Thesis/data/euler/dalia_preprocessed")

    dataset = DaLiADataset(path, [1, 2, 3], False, True)
    for x, y in dataset:
        print(x.shape)
        print(y.shape)
        break

    #  module = DaLiADataModule(str(path))
    # module.setup("fit")
    import pdb

    pdb.set_trace()

    """
    for mode in tqdm(modes):
        for b in use_heart_rate:
            dataset = DaLiADataset(
                path,
                "train",
                use_heart_rate=False,
                look_back_window=10,
                prediction_window=5,
            )
            indices = np.random.choice(len(dataset), size=100, replace=False)
            for idx in indices:
                look_back_window, prediction_window = dataset[idx]

    """
