import pandas as pd
import numpy as np
import torch

from torch.utils.data import Dataset

from src.datasets.utils import BaseDataModule


def ptbxl_load_data(datadir: str):
    raise NotImplementedError()


def get_ids(folds: list[int], summary: pd.DataFrame):
    mapping = {ecg_id: i for i, ecg_id in enumerate(summary["ecg_id"].astype(str))}
    ecg_ids = summary[summary["strat_fold"].isin(folds)]["ecg_id"].astype(str)
    return [mapping[ecg_id] for ecg_id in ecg_ids]


class PTBXLDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        data: np.ndarray,
        use_disease: bool = False,
    ):
        self.n_records = len(data)
        self.data = data
        assert data.shape == (self.n_records, 1000, 12)
        self.use_disease = use_disease
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

    def __len__(self) -> int:
        return self.n_records * (1000 - self.window + 1)

    def __getitem__(self, idx: int) -> torch.Tensor:
        participant_idx = idx // (1000 - self.window + 1)
        window_pos = idx % (1000 - self.window + 1)

        look_back_window = self.data[participant_idx][
            window_pos : window_pos + self.look_back_window, :
        ]
        prediction_window = self.data[participant_idx][
            window_pos + self.look_back_window : window_pos + self.window, :
        ]

        if self.use_disease:
            disease_vec = np.repeat(
                self.disease[participant_idx][np.newaxis, :],
                self.look_back_window,
                axis=0,
            )
            look_back_window = np.concatenate((look_back_window, disease_vec), axis=1)

        look_back_window = torch.from_numpy(look_back_window).float()
        prediction_window = torch.from_numpy(prediction_window).float()

        return look_back_window, prediction_window


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
        use_heart_rate: bool = False,
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

        self.data = np.load(data_dir + "ptbxl.npy")

        summary = pd.read_csv(data_dir + "ptbxl_database.csv")
        # TODO add exogenous variables

        self.train_ids = get_ids(train_folds, summary)
        self.val_ids = get_ids(val_folds, summary)
        self.test_ids = get_ids(test_folds, summary)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.train_ids],
                use_disease=self.use_activity_info,
            )
            self.val_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.val_ids],
                use_disease=self.use_activity_info,
            )
        if stage == "test":
            self.test_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.test_ids],
                use_disease=self.use_activity_info,
            )
