import numpy as np
import torch
import lightning as L

from torch.utils.data import Dataset, DataLoader


class WildPPGDataset(Dataset):
    def __init__(
        self,
        datadir: str,
        use_heart_rate: bool = False,
        look_back_window: int = 320,
        prediction_window: int = 128,
        participants: list[str] = [
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
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        self.arrays = []

        prefix = "WildPPG_Part_"
        for participant in participants:
            data = np.load(datadir + prefix + participant + ".npz")
            activity = data["activity"]
            if use_heart_rate:
                self.arrays.append(data["ecg"])
            else:
                self.arrays.append(data["ppg"])

        self.lengths = [len(arr) - self.window + 1 for arr in self.arrays]
        self.cumulative_lengths = np.cumsum([0] + self.lengths)
        self.total_length = self.cumulative_lengths[-1]

    def __len__(self) -> int:
        return self.total_length

    def __getitem__(self, idx: int) -> torch.Tensor:
        file_idx = np.searchsorted(self.cumulative_lengths, idx, side="right") - 1
        index = idx - self.cumulative_lengths[file_idx]
        window = self.arrays[file_idx][index : (index + self.window)]
        look_back_window = torch.Tensor(window[: self.look_back_window])
        prediction_window = torch.Tensor(window[self.look_back_window :])

        return look_back_window.float(), prediction_window.float()


class WildPPGDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        use_heart_rate: bool = False,
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
    ):
        super().__init__()
        self.data_dir = data_dir
        self.use_heart_rate = use_heart_rate
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

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
            )
            self.val_dataset = WildPPGDataset(
                self.data_dir,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = WildPPGDataset(
                self.data_dir,
                self.use_heart_rate,
                self.look_back_window,
                self.prediction_window,
                self.test_participants,
            )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size)


if __name__ == "__main__":
    from tqdm import tqdm

    path = "C:/Users/cleme/ETH/Master/Thesis/data/WildPPG/data/"
    dataset = WildPPGDataset(path)
    for i in tqdm(range(len(dataset))):
        look_back_window, prediction_window = dataset[i]
        break
