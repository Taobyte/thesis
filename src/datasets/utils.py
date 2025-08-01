import numpy as np
import random
import torch
import lightning as L

from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from numpy.typing import NDArray
from einops import rearrange
from typing import Tuple, Optional

from src.normalization import global_z_norm


class BaseDataModule(L.LightningDataModule):
    look_back_channel_dim: int
    target_channel_dim: int
    use_static_features: bool
    use_dynamic_features: bool
    static_exogenous_variables: int
    dynamic_exogenous_variables: int
    local_norm_channels: int

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        name: str,
        freq: int,
        look_back_window: int,
        prediction_window: int,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 6,
        look_back_channel_dim: int = 1,
        normalization: str = "global",
        use_only_exo: bool = False,
        use_perfect_info: bool = False,
    ):
        super().__init__()

        self.name = name
        self.freq = freq
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        # Experiment Flags
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.use_only_exo = use_only_exo
        self.use_perfect_info = use_perfect_info

        self.dynamic_exogenous_variables = dynamic_exogenous_variables
        self.static_exogenous_variables = static_exogenous_variables

        self.target_channel_dim = target_channel_dim
        self.look_back_channel_dim = look_back_channel_dim

        self.local_norm_channels = (
            target_channel_dim + dynamic_exogenous_variables
            if use_dynamic_features
            else target_channel_dim
        )

        self.normalization = normalization

        self.train_dataset: Optional[Dataset[NDArray[np.float32]]] = None
        self.val_dataset = None
        self.test_dataset = None

    def postprocess_batch(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        B, _, C = look_back_window.shape
        device = look_back_window.device
        if self.normalization == "global":
            mean, std = self.train_dataset.mean, self.train_dataset.std
            mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
            std = torch.tensor(std).reshape(1, 1, -1).to(device).float()
            input = global_z_norm(look_back_window, self.local_norm_channels, mean, std)
            output = global_z_norm(
                prediction_window, self.local_norm_channels, mean, std
            )
        elif self.normalization == "difference":
            pad = torch.zeros(B, 1, C)
            input = torch.cat([pad, torch.diff(look_back_window, dim=1)], dim=1)
            last_value = look_back_window[:, -1:, :]
            output = torch.diff(
                torch.cat([last_value, prediction_window], dim=1), dim=1
            )
        else:
            input = look_back_window
            output = prediction_window

        if self.use_only_exo:
            assert self.use_dynamic_features
            input = input[:, :, self.target_channel_dim :]
        elif self.use_perfect_info:
            ex_concat = torch.concat(
                [
                    input[:, :, self.target_channel_dim :],
                    output[:, :, self.target_channel_dim :],
                ],
                dim=1,
            )
            perf_info_channel = ex_concat[:, -self.look_back_window :, :]
            input = torch.concat((input, perf_info_channel), dim=-1)

        output = output[:, :, : self.target_channel_dim]

        return look_back_window, input, output

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn_with_postprocessing,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn_with_postprocessing,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, len(self.test_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn_with_postprocessing,
        )

    def collate_fn_with_postprocessing(
        self, batch: Tuple[NDArray[np.float32], NDArray[np.float32]]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        look_back, prediction = zip(*batch)
        look_back = torch.stack(look_back)
        prediction = torch.stack(prediction)
        return self.postprocess_batch(look_back, prediction)

    # used for Gaussian Process model & shap calculation
    def get_inducing_points(
        self, num_inducing: int = 500, strategy: str = "random", mode: str = "train"
    ) -> torch.Tensor:
        # Ensure the train/test dataset is ready
        if mode == "train":
            self.setup(stage="fit")
            assert self.train_dataset is not None, "Train dataset is not initialized."
        else:
            self.setup(stage="test")
            assert self.test_dataset is not None, "Train dataset is not initialized."
        dataset = self.train_dataset if mode == "train" else self.test_dataset
        dataset_length = len(dataset)

        assert dataset_length >= num_inducing, (
            f"Cannot sample {num_inducing} inducing points from dataset of size {dataset_length}"
        )
        if strategy == "random":
            indices = random.sample(range(dataset_length), num_inducing)
            inducing_points = [dataset[i][0] for i in indices]

            inducing_points_tensor = torch.stack(inducing_points, dim=0).requires_grad_(
                True
            )
            # inducing_points_tensor = rearrange(inducing_points, "B T C -> B (T C)")
        else:
            # use KMeans clustering
            from sklearn.cluster import KMeans

            train_x, _ = (
                self.get_train_dataset() if mode == "train" else self.get_test_dataset()
            )
            train_x = rearrange(train_x, "B T C -> B (T C)")
            kmeans = KMeans(n_clusters=num_inducing).fit(train_x)
            # TODO: use the nearest data point to the cluster instead
            inducing_points_tensor = torch.tensor(
                kmeans.cluster_centers_, requires_grad=True
            )
            inducing_points_tensor = rearrange(
                inducing_points_tensor, "B (T C) -> B T C", C=self.look_back_channel_dim
            )

        return inducing_points_tensor

    def get_train_dataset_length(self):
        self.setup(stage="fit")

        assert self.train_dataset is not None, "Train dataset is not initialized."

        print(
            f"Length of train dataset for Gaussian Process: {len(self.train_dataset)}"
        )

        return len(self.train_dataset)

    def _get_dataset(
        self, mode: str = "train"
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """
        return z-normalized dataset
        """
        self.setup(stage="fit")
        if mode == "train":
            dataloader = self.train_dataloader()
        elif mode == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()

        lbws = []
        pws = []
        for _, look_back_window_norm, prediction_window_norm in dataloader:
            look_back_window_norm = look_back_window_norm.detach().cpu().numpy()
            prediction_window_norm = prediction_window_norm.detach().cpu().numpy()
            lbws.append(look_back_window_norm)
            pws.append(prediction_window_norm)

        lbws_dataset = np.concatenate(lbws, axis=0)
        pws_dataset = np.concatenate(pws, axis=0)

        lbw_gb = lbws_dataset.nbytes / (1024**3)
        pw_gb = pws_dataset.nbytes / (1024**3)
        total_gb = lbw_gb + pw_gb

        print(f"[{mode}] look_back_window: {lbws_dataset.shape}, {lbw_gb:.5f} GB")
        print(f"[{mode}] prediction_window: {pws_dataset.shape}, {pw_gb:.5f} GB")
        print(f"[{mode}] total size: {total_gb:.5f} GB")

        return lbws_dataset, pws_dataset

    def get_train_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("train")

    def get_val_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("val")

    def get_test_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("test")

    def get_numpy_normalized(self, type: str = "train") -> list[NDArray[np.float32]]:
        assert type in ["train", "val"]
        if type == "train":
            data = self.train_dataset.data
        elif type == "val":
            data = self.val_dataset.data
        normalized_data: list[NDArray[np.float32]] = []
        for s in data:
            if self.normalization == "global":
                mean = self.train_dataset.mean
                std = self.train_dataset.std
                normalized = (s - mean) / (std + 1e-6)
            elif self.normalization == "difference":
                normalized = np.diff(s, axis=0)
            else:
                normalized = s
            normalized_data.append(normalized)

        return normalized_data
