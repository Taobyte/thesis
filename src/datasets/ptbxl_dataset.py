import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

from src.datasets.utils import BaseDataModule


def ptb_load_data(datadir: str):
    return None


class PTBXLDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        X: np.ndarray,
        disease: np.ndarray,
        use_disease: bool = False,
    ):
        self.X = X
        self.disease = disease
        self.use_disease = use_disease
        self.n_participants = len(disease)
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

    def __len__(self) -> int:
        return self.n_participants * (1000 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        participant_idx = idx // (1000 - self.window + 1)
        window_pos = idx % (1000 - self.window + 1)

        look_back_window = self.X[participant_idx][
            window_pos : window_pos + self.look_back_window, :
        ]
        prediction_window = self.X[participant_idx][
            window_pos + self.look_back_window : window_pos + self.window, :
        ]

        if self.use_disease:
            disease_vec = np.repeat(
                self.disease[participant_idx][np.newaxis, :],
                self.look_back_window,
                axis=0,
            )
            look_back_window = np.concatenate((look_back_window, disease_vec), axis=1)

        return torch.from_numpy(look_back_window).float(), torch.from_numpy(
            prediction_window
        ).float()


class PTBXLDataModule(BaseDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int = 32,
        look_back_window: int = 128,
        prediction_window: int = 64,
        use_disease: bool = False,
        train_folds: list[int] = [1, 2, 3, 4, 5],
        val_folds: list[int] = [6, 7],
        test_folds: list[int] = [8, 9, 10],
        freq: int = 100,
        name: str = "ptbxl",
        num_workers: int = 0,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            name=name,
            freq=freq,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_activity_info=use_disease,  # we feed in disease information for Chapman
        )

        self.train_folds = train_folds
        self.val_folds = val_folds
        self.test_folds = test_folds

        df = pd.read_csv(data_dir + "/ptbxl_database.csv")

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_train,
                self.y_train,
                use_disease=self.use_activity_info,
            )
            self.val_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_val,
                self.y_val,
                use_disease=self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.X_test,
                self.y_test,
                use_disease=self.use_activity_info,
            )
