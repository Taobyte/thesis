import pandas as pd
import numpy as np
import torch
import ast

from torch.utils.data import Dataset

from src.datasets.utils import BaseDataModule


class PTBXLDataset(Dataset):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        data: np.ndarray,
        static_features: np.ndarray,
        use_static_features: bool = False,
    ):
        self.n_records = len(data)
        self.data = data
        self.static_features = static_features

        assert data.shape == (self.n_records, 1000, 12)

        self.use_static_features = use_static_features
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window = look_back_window + prediction_window

        assert self.window <= 1000

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

        if self.use_static_features:
            disease_vec = np.repeat(
                self.static_features[participant_idx][np.newaxis, :],
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
        train_folds: list[int] = [1, 2, 3, 4, 5],
        val_folds: list[int] = [6, 7],
        test_folds: list[int] = [8, 9, 10],
        freq: int = 100,
        name: str = "ptbxl",
        num_workers: int = 0,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 6,
        look_back_channel_dim: int = 1,
    ):
        super().__init__(
            data_dir=data_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            name=name,
            freq=freq,
            look_back_window=look_back_window,
            prediction_window=prediction_window,
            use_dynamic_features=use_dynamic_features,
            use_static_features=use_static_features,
            target_channel_dim=target_channel_dim,
            dynamic_exogenous_variables=dynamic_exogenous_variables,
            static_exogenous_variables=static_exogenous_variables,
            look_back_channel_dim=look_back_channel_dim,
        )

        self.data = np.load(data_dir + "ptbxl.npy")

        summary = pd.read_csv(data_dir + "ptbxl_database.csv")
        scp_statements = pd.read_csv(data_dir + "scp_statements.csv")

        scp_mapping = dict(
            zip(scp_statements["Unnamed: 0"], scp_statements["diagnostic_class"])
        )

        diseases = scp_statements["Unnamed: 0"].iloc[:44].tolist()
        summary["filtered"] = summary["scp_codes"].apply(
            lambda x: {k: v for k, v in ast.literal_eval(x).items() if k in diseases}
        )

        summary["scp_code_processed"] = summary["filtered"].apply(
            lambda x: scp_mapping[max(x, key=x.get)] if len(x) > 0 else "UNKNOWN"
        )

        summary["age"] = (summary["age"] - summary["age"].mean()) / (
            summary["age"].std() + 1e-8
        )

        def get_ids(folds: list[int], summary: pd.DataFrame) -> list[int]:
            mapping = {
                ecg_id: i for i, ecg_id in enumerate(summary["ecg_id"].astype(str))
            }
            ecg_ids = summary[summary["strat_fold"].isin(folds)]["ecg_id"].astype(str)
            return [mapping[ecg_id] for ecg_id in ecg_ids]

        one_hot = pd.get_dummies(summary["scp_code_processed"], prefix="disease") * 1
        self.static_features = pd.concat((summary[["ecg_id", "age"]], one_hot), axis=1)

        self.train_ids = get_ids(train_folds, summary)
        self.val_ids = get_ids(val_folds, summary)
        self.test_ids = get_ids(test_folds, summary)

    def get_static_features(self, ids: list[int]) -> np.ndarray:
        df = self.static_features.copy()
        df = df[df["ecg_id"].isin(ids)]
        df = df.sort_values("ecg_id")
        df = df.drop(columns=["ecg_id"])
        return df.values

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.train_ids],
                self.get_static_features(self.train_ids),
                use_static_features=self.use_static_features,
            )
            self.val_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.val_ids],
                self.get_static_features(self.val_ids),
                use_static_features=self.use_static_features,
            )
        if stage == "test":
            self.test_dataset = PTBXLDataset(
                self.look_back_window,
                self.prediction_window,
                self.data[self.test_ids],
                self.get_static_features(self.test_ids),
                use_static_features=self.use_static_features,
            )
