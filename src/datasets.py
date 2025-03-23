import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from enum import Enum
from pathlib import Path
from typing import Tuple


import pdb


class Mode(Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class DaLiADataset(Dataset):
    def __init__(
        self,
        path: Path,
        mode: Mode = Mode.TRAIN,
        look_back_window: int = 320,
        prediction_window: int = 128,
        train: list = [1, 2, 3, 4, 5, 6, 7, 8, 9],
        val: list = [10, 11, 12],
        test: list = [13, 14, 15],
    ):
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window
        if mode == Mode.TRAIN:
            cases = train
        elif mode == Mode.VAL:
            cases = val
        else:
            cases = test
        self.cases = []
        self.total_length = 0
        for i in cases:
            label = "S" + str(i)
            case_path = Path(path) / label / (label + "_E4") / "BVP.csv"
            df = pd.read_csv(case_path)
            self.cases.append(df)
            self.total_length += len(df)

    def __len__(self) -> int:
        return self.total_length - len(self.cases) * (
            self.look_back_window + self.prediction_window - 1
        )

    def _idx_to_case(self, idx: int) -> Tuple[int, int]:
        current = len(self.cases[0]) - self.look_back_window - self.prediction_window
        for i in range(len(self.cases)):
            if idx <= current:
                return i, (current - idx)
            else:
                current += (
                    len(self.cases[i + 1])
                    - self.look_back_window
                    - self.prediction_window
                    + 1
                )

    def __getitem__(self, idx: int) -> torch.Tensor:
        case_index, index = self._idx_to_case(idx)

        window = self.cases[case_index][index : (index + self.window)]
        x = torch.Tensor(window.iloc[: self.look_back_window, 0].values)
        y = torch.Tensor(window.iloc[self.look_back_window :, 0].values)

        return x, y


if __name__ == "__main__":
    from tqdm import tqdm

    path = Path("C:/Users/cleme/ETH/Master/Thesis/data/DaLiA/data/PPG_FieldStudy")
    dataset = DaLiADataset(path, train=[1], val=[2])
    for i in tqdm(range(len(dataset))):
        x, y = dataset[i]
