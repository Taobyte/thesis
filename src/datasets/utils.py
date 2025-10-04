import numpy as np
import random
import torch
import lightning as L

from torch.nn.utils.rnn import pad_sequence
from torch import Tensor
from torch.utils.data import DataLoader
from numpy.typing import NDArray
from einops import rearrange
from typing import Tuple, List

from src.normalization import global_z_norm, min_max_norm, local_z_norm_numpy


class BaseDataModule(L.LightningDataModule):
    look_back_channel_dim: int
    target_channel_dim: int
    use_static_features: bool
    use_dynamic_features: bool
    static_exogenous_variables: int
    dynamic_exogenous_variables: int

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        num_workers: int,
        name: str,
        look_back_window: int,
        prediction_window: int,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        target_channel_dim: int = 1,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 6,
        look_back_channel_dim: int = 1,
        normalization: str = "global",
        test_local: bool = False,
        train_frac: float = 0.7,
        val_frac: float = 0.1,
        return_whole_series: bool = False,
        local_norm: str = "none",
        local_norm_endo_only: bool = False,
    ):
        super().__init__()

        self.name = name
        self.data_dir = data_dir

        self.batch_size = batch_size
        self.num_workers = num_workers

        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

        # Experiment Flags
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features

        self.dynamic_exogenous_variables = dynamic_exogenous_variables
        self.static_exogenous_variables = static_exogenous_variables

        self.target_channel_dim = target_channel_dim
        self.look_back_channel_dim = look_back_channel_dim

        self.return_whole_series = return_whole_series

        self.normalization = normalization
        self.local_norm = local_norm
        self.local_norm_endo_only = local_norm_endo_only

        self.test_local = test_local
        self.train_frac = train_frac
        self.val_frac = val_frac

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def postprocess_batch(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor]:
        device = look_back_window.device
        if self.normalization == "global":
            mean, std = self.train_dataset.mean, self.train_dataset.std
            mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
            std = torch.tensor(std).reshape(1, 1, -1).to(device).float()
            input = global_z_norm(look_back_window, mean, std)
            output = global_z_norm(prediction_window, mean, std)
        elif self.normalization == "minmax":
            min, max = self.train_dataset.min, self.train_dataset.max
            min = torch.tensor(min).reshape(1, 1, -1).to(device).float()
            max = torch.tensor(max).reshape(1, 1, -1).to(device).float()
            input = min_max_norm(look_back_window, min, max)
            output = min_max_norm(prediction_window, min, max)
        else:
            input = look_back_window
            output = prediction_window

        output = output[:, :, : self.target_channel_dim]
        return input, output

    def postprocess_series(self, series: Tensor) -> Tensor:
        B, _, C = series.shape
        device = series.device
        if self.normalization == "global":
            mean, std = self.train_dataset.mean, self.train_dataset.std
            mean = torch.tensor(mean).reshape(1, 1, -1).to(device).float()
            std = torch.tensor(std).reshape(1, 1, -1).to(device).float()
            input = global_z_norm(series, mean, std)
        elif self.normalization == "minmax":
            min, max = self.train_dataset.min, self.train_dataset.max
            min = torch.tensor(min).reshape(1, 1, -1).to(device).float()
            max = torch.tensor(max).reshape(1, 1, -1).to(device).float()
            input = min_max_norm(series, min, max)
        else:
            input = series

        x = input.clone()
        min_channels = self.target_channel_dim if self.local_norm_endo_only else C
        means = stdev = None

        if self.local_norm == "local_z":
            means = x.mean(1, keepdim=True).detach()
            var = x.var(1, keepdim=True, unbiased=False).detach()
            stdev = torch.sqrt(var) + 1e-5
            x[:, :, :min_channels] = (
                x[:, :, :min_channels] - means[:, :, :min_channels]
            ) / stdev[:, :, :min_channels]

        elif self.local_norm == "difference":
            x[:, :, :min_channels] = torch.diff(
                x[:, :, :min_channels], dim=1, prepend=x[:, :1, :min_channels]
            )

        return x

    def collate_fn_with_postprocessing(
        self, batch: Tuple[Tensor, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        look_back, prediction = zip(*batch)
        look_back = torch.stack(look_back)
        prediction = torch.stack(prediction)
        return self.postprocess_batch(look_back, prediction)

    def collate_fn_series(self, series: List[Tensor]):
        lengths = torch.tensor([s.size(0) for s in series], dtype=torch.long)
        padded = pad_sequence(series, batch_first=True, padding_value=0.0)
        return self.postprocess_series(padded), lengths

    def train_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            collate_fn=self.collate_fn_with_postprocessing
            if not self.return_whole_series
            else self.collate_fn_series,
        )

    def val_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.val_dataset,
            batch_size=min(self.batch_size, len(self.val_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn_with_postprocessing
            if not self.return_whole_series
            else self.collate_fn_series,
        )

    def test_dataloader(self) -> DataLoader[tuple[Tensor, Tensor, Tensor]]:
        return DataLoader(
            self.test_dataset,
            batch_size=min(self.batch_size, len(self.test_dataset)),
            num_workers=self.num_workers,
            shuffle=False,
            collate_fn=self.collate_fn_with_postprocessing,
        )

    # used for Gaussian Process model
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
        if dataset_length < num_inducing:
            print(
                f"Number of inducing points {num_inducing} are larger than dataset length {dataset_length}"
            )
            print("Setting num_inducing = dataset_length")
            num_inducing = dataset_length

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

        print(f"Length of train dataset: {len(self.train_dataset)}")

        return len(self.train_dataset)

    def _get_dataset(
        self, mode: str = "train"
    ) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        self.setup(stage="fit")
        if mode == "train":
            dataloader = self.train_dataloader()
        elif mode == "val":
            dataloader = self.val_dataloader()
        else:
            dataloader = self.test_dataloader()

        lbws = []
        pws = []
        for look_back_window_norm, prediction_window_norm in dataloader:
            look_back_window_norm = look_back_window_norm.detach().cpu().numpy()
            prediction_window_norm = prediction_window_norm.detach().cpu().numpy()
            lbws.append(look_back_window_norm)
            pws.append(prediction_window_norm)

        lbws_dataset = np.concatenate(lbws, axis=0)
        pws_dataset = np.concatenate(pws, axis=0)

        lbw_gb = lbws_dataset.nbytes / (1024**3)
        pw_gb = pws_dataset.nbytes / (1024**3)
        total_gb = lbw_gb + pw_gb

        # print(f"[{mode}] look_back_window: {lbws_dataset.shape}, {lbw_gb:.5f} GB")
        # print(f"[{mode}] prediction_window: {pws_dataset.shape}, {pw_gb:.5f} GB")
        # print(f"[{mode}] total size: {total_gb:.5f} GB")

        return lbws_dataset, pws_dataset

    def get_train_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("train")

    def get_val_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("val")

    def get_test_dataset(self) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        return self._get_dataset("test")

    def get_numpy_dataset(
        self,
    ) -> Tuple[
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
        NDArray[np.float32],
    ]:
        X_train, y_train = self.get_train_dataset()
        X_val, y_val = self.get_val_dataset()

        B, T, C = X_train.shape
        channels = self.target_channel_dim if self.local_norm_endo_only else C
        if self.local_norm == "local_z":
            X_train, mean, std = local_z_norm_numpy(X_train, channels=channels)
            y_train, _, _ = local_z_norm_numpy(y_train, mean, std, channels=channels)
            X_val, val_mean, val_std = local_z_norm_numpy(X_val, channels=channels)
            y_val, _, _ = local_z_norm_numpy(
                y_val, val_mean, val_std, channels=channels
            )
        elif self.local_norm == "difference":
            x_diffed = np.diff(X_train, axis=1, prepend=X_train[:, :1, :])
            y_diffed = np.diff(
                y_train, axis=1, prepend=X_train[:, -1:, : self.target_channel_dim]
            )
            X_train[:, :, :channels] = x_diffed[:, :, :channels]
            y_train[:, :, : self.target_channel_dim] = y_diffed[
                :, :, : self.target_channel_dim
            ]

            x_diffed = np.diff(X_val, axis=1, prepend=X_val[:, :1, :])
            y_diffed = np.diff(
                y_val, axis=1, prepend=X_val[:, -1:, : self.target_channel_dim]
            )
            X_val[:, :, :channels] = x_diffed[:, :, :channels]
            y_val[:, :, : self.target_channel_dim] = y_diffed[
                :, :, : self.target_channel_dim
            ]

        return X_train, y_train, X_val, y_val
