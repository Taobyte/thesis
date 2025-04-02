import torch
import lightning as L
from torch.utils.data import DataLoader, Dataset


class IEEEDataset(Dataset):
    def __init__(self, datadir: str):
        self.datadir = datadir

    def __getitem__(self, idx: int) -> torch.Tensor:
        pass


class IEEEDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.train_participants,
            )
            self.val_dataset = IEEEDataset(
                self.data_dir,
                self.look_back_window,
                self.prediction_window,
                self.val_participants,
            )
        if stage == "test":
            self.test_dataset = IEEEDataset(
                self.data_dir,
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
