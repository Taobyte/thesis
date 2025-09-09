import torch
import torch.nn as nn
import gpytorch
import gpytorch.settings

from torch import Tensor
from einops import rearrange
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    LMCVariationalStrategy,
)
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from gpytorch.distributions import MultivariateNormal

from typing import Tuple, Any

from src.models.utils import BaseLightningModule


class TimeSeriesFeatureExtractor(nn.Module):
    def __init__(
        self,
        look_back_channel_dim: int = 1,
        hidden_dim: int = 8,
        out_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(look_back_channel_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.net(x)  # -> (B, hidden_dim, 1)
        x = x.squeeze(-1)  # -> (B, hidden_dim)
        return self.fc(x)  # -> (B, out_dim)


# -----------------------------------------------------------------------------
# Standard Gaussian Process
# -----------------------------------------------------------------------------


class GPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points: Tensor,
        train_dataset_length: int,
        num_latents: int = 3,
        num_tasks: int = 3,
        kernel: str = "rbf",
        use_linear_trend: bool = False,
        periodic_type: str = "",
        use_feature_extractor: bool = False,
        out_dim: int = 16,
    ):
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.train_dataset_length = train_dataset_length
        # we first have to rearrange the inducing points to have only two dimensions
        if not use_feature_extractor:
            inducing_points = rearrange(inducing_points, "B T C -> B (T C)")
            inducing_points = inducing_points.unsqueeze(0).expand(num_latents, -1, -1)

        else:
            B, _, _ = inducing_points.shape

            """
            variational_distribution = (
                gpytorch.variational.CholeskyVariationalDistribution(
                    num_inducing_points=B, batch_shape=torch.Size([num_latents])
                )
            )
            variational_strategy = gpytorch.variational.LMCVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=2,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_tasks,
                num_latents=num_latents,
                latent_dim=-1,
            )
            else:
            """
            inducing_points = torch.randn((num_latents, B, out_dim))

        # We have to mark the CholeskyVariationalDistribution as batch
        # so that we learn a variational distribution for each task
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

        # We have to wrap the VariationalStrategy in a LMCVariationalStrategy
        # so that the output will be a MultitaskMultivariateNormal rather than a batch output
        variational_strategy = LMCVariationalStrategy(
            gpytorch.variational.VariationalStrategy(
                self,
                inducing_points,
                variational_distribution,
                learn_inducing_locations=True,
            ),
            num_tasks=num_tasks,
            num_latents=num_latents,
            latent_dim=-1,
        )
        super().__init__(variational_strategy)

        # The mean and covariance modules should be marked as batch
        # so we learn a different set of hyperparameters
        self.mean_module = gpytorch.means.ConstantMean(
            batch_shape=torch.Size([num_latents]),
        )

        if kernel == "rbf":
            kernel = gpytorch.kernels.RBFKernel(batch_shape=torch.Size([num_latents]))
        elif kernel == "matern":
            kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, batch_shape=torch.Size([num_latents])
            )
        elif kernel == "rq":
            kernel = gpytorch.kernels.RQKernel(batch_shape=torch.Size([num_latents]))
        else:
            raise NotImplementedError()

        #         elif kernel == "sm":
        #             kernel = gpytorch.kernels.SpectralMixtureKernel(
        #                 num_mixtures=12,
        #                 batch_shape=torch.Size([num_latents]),
        #                 ard_num_dims=inducing_points.shape[-1],
        #             )
        if periodic_type == "multiplicative":
            periodic_kernel = gpytorch.kernels.PeriodicKernel(
                batch_shape=torch.Size([num_latents])
            )
            kernel *= periodic_kernel
        elif periodic_type == "additive":
            periodic_kernel = gpytorch.kernels.PeriodicKernel(
                batch_shape=torch.Size([num_latents])
            )
            kernel += periodic_kernel

        if use_linear_trend:
            kernel += gpytorch.kernels.LinearKernel(
                batch_shape=torch.Size([num_latents])
            )

        if kernel in ["sm"]:
            self.covar_module = (
                kernel  # spectralMixture kernel should not be combined with scalekernel
            )
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                kernel, batch_shape=torch.Size([num_latents])
            )

    def forward(self, x: Tensor) -> MultivariateNormal:
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class Model(gpytorch.Module):
    def __init__(
        self,
        inducing_points: torch.Tensor,
        train_dataset_length: int,
        num_latents: int = 3,
        num_tasks: int = 3,
        kernel: str = "rbf",
        use_linear_trend: bool = False,
        periodic_type: str = "",
        look_back_channel_dim: int = 1,
        hidden_dim: int = 8,
        out_dim: int = 16,
        dropout: float = 0.2,
        use_feature_extractor: bool = False,
        grid_bounds: Tuple[float, float] = (-10.0, 10.0),
    ):
        super(Model, self).__init__()

        self.num_tasks = num_tasks
        self.train_dataset_length = train_dataset_length
        self.use_feature_extractor = use_feature_extractor

        self.feature_extractor = (
            TimeSeriesFeatureExtractor(
                look_back_channel_dim=look_back_channel_dim,
                hidden_dim=hidden_dim,
                out_dim=out_dim,
                dropout=dropout,
            )
            if use_feature_extractor
            else nn.Identity()
        )

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            grid_bounds[0], grid_bounds[1]
        )

        self.gp_layer = GPModel(
            inducing_points,
            train_dataset_length,
            num_latents=num_latents,
            num_tasks=num_tasks,
            kernel=kernel,
            use_linear_trend=use_linear_trend,
            periodic_type=periodic_type,
            use_feature_extractor=use_feature_extractor,
            out_dim=out_dim,
        )

    def forward(self, x: Tensor):
        features = self.feature_extractor(x)
        if self.use_feature_extractor:
            features = self.scale_to_bounds(features)
        res = self.gp_layer(features)
        return res


class GaussianProcess(BaseLightningModule):
    def __init__(
        self,
        model: Model,
        use_feature_extractor: bool = False,
        learning_rate: float = 0.001,
        jitter: float = 1e-6,
        use_norm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.jitter = jitter
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=model.num_tasks)

        self.mll = VariationalELBO(
            self.likelihood, self.model.gp_layer, num_data=model.train_dataset_length
        )
        self.mae_loss = torch.nn.L1Loss()

        self.use_feature_extractor = use_feature_extractor
        self.use_norm = use_norm

    def model_specific_forward(
        self, look_back_window: torch.Tensor
    ) -> Tuple[Tensor, Tensor]:
        T = self.trainer.datamodule.prediction_window
        if not self.use_feature_extractor:
            look_back_window = rearrange(
                look_back_window, "B T C -> B (T C)"
            )  # GPytorch assumes flattened channels
        with gpytorch.settings.cholesky_jitter(self.jitter):
            preds = self.model(look_back_window)

        mean = rearrange(preds.mean, "B (T C) -> B T C", T=T)
        std = rearrange(preds.stddev, "B (T C) -> B T C", T=T)

        return mean, std

    def _shared_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor]:
        if not self.use_feature_extractor:
            _, _, C = prediction_window.shape
            look_back_window = rearrange(
                look_back_window, "B T C -> B (T C)"
            )  # GPytorch assumes flattened channels
        with gpytorch.settings.cholesky_jitter(self.jitter):
            preds = self.model(look_back_window)
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")
        loss = -self.mll(preds, prediction_window)
        mae_loss = self.mae_loss(preds.mean, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
