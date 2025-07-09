import numpy as np
import random
import torch
import lightning as L

from einops import rearrange
from torch.utils.data import DataLoader
from typing import Tuple

from src.normalization import local_z_norm, global_z_norm


class BaseDataModule(L.LightningDataModule):
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
        normalization: str = "local",
    ):
        super().__init__()

        self.name = name
        self.freq = freq
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features

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

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, len(self.test_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
        )

    # used for Gaussian Process model & shap calculation
    def get_inducing_points(
        self, num_inducing: int = 500, strategy: str = "random", mode: str = "train"
    ) -> torch.Tensor:
        """
        Selects a subset of data points to serve as inducing points for sparse Gaussian Process
        (GP) models or other methods that require a representative subset of the input space.

        This function supports two strategies: random sampling and K-Means clustering.

        Args:
            num_inducing: The desired number of inducing points.
            strategy: The strategy to use for selecting inducing points.
                      Supported values are "random" for random sampling or "kmeans" for K-Means clustering.
            mode: Specifies which dataset to draw inducing points from.
                  Can be "train" for the training set or "test" for the test set.

        Returns:
            A `torch.Tensor` of shape `(num_inducing, Time, Channels)` containing the selected
            inducing points. Each channel is z-normalized.
        """

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

    def _get_dataset(self, mode: str = "train") -> Tuple[np.ndarray, np.ndarray]:
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
        for look_back_window, prediction_window in dataloader:
            if self.normalization == "local":
                look_back_window_norm, mean, std = local_z_norm(
                    look_back_window, self.local_norm_channels
                )
                prediction_window_norm, _, _ = local_z_norm(
                    prediction_window, self.local_norm_channels, mean, std
                )
            elif self.normalization == "global":
                look_back_window_norm = global_z_norm(
                    look_back_window,
                    self.local_norm_channels,
                    self.train_dataset.mean,
                    self.train_dataset.std,
                )
                prediction_window_norm = global_z_norm(
                    prediction_window,
                    self.local_norm_channels,
                    self.train_dataset.mean,
                    self.train_dataset.std,
                )

            elif self.normalization == "none":
                look_back_window_norm = look_back_window
                prediction_window_norm = prediction_window
            else:
                raise NotImplementedError()
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

    def get_train_dataset(self) -> np.ndarray:
        return self._get_dataset("train")

    def get_val_dataset(self) -> np.ndarray:
        return self._get_dataset("val")

    def get_test_dataset(self) -> np.ndarray:
        return self._get_dataset("test")
