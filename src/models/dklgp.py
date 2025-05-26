import gpytorch
import torch
import math

import torch.nn as nn

from torch.optim.lr_scheduler import MultiStepLR
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from einops import rearrange

from src.models.utils import BaseLightningModule

# -----------------------------------------------------------------------------
# SVDKL (Stochastic Variational Deep Kernel Learning)
# -----------------------------------------------------------------------------


class TimeSeriesFeatureExtractor(nn.Module):
    def __init__(self, input_channels: int, hidden_dim: int = 64, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),  # (B, hidden_dim, 1)
        )
        self.fc = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T) for Conv1d
        x = self.net(x)  # -> (B, hidden_dim, 1)
        x = x.squeeze(-1)  # -> (B, hidden_dim)
        return self.fc(x)  # -> (B, out_dim)


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
    def __init__(self, num_dim, grid_bounds=(-10.0, 10.0), grid_size=64):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=grid_size, batch_shape=torch.Size([num_dim])
        )

        # Our base variational strategy is a GridInterpolationVariationalStrategy,
        # which places variational inducing points on a Grid
        # We wrap it with a IndependentMultitaskVariationalStrategy so that our output is a vector-valued GP
        variational_strategy = (
            gpytorch.variational.IndependentMultitaskVariationalStrategy(
                gpytorch.variational.GridInterpolationVariationalStrategy(
                    self,
                    grid_size=grid_size,
                    grid_bounds=[grid_bounds],
                    variational_distribution=variational_distribution,
                ),
                num_tasks=num_dim,
            )
        )
        super().__init__(variational_strategy)

        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                lengthscale_prior=gpytorch.priors.SmoothedBoxPrior(
                    math.exp(-1), math.exp(1), sigma=0.1, transform=torch.exp
                )
            )
        )
        self.mean_module = gpytorch.means.ConstantMean()
        self.grid_bounds = grid_bounds

    def forward(self, x):
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLModel(torch.nn.Module):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        inducing_points,
        train_dataset_length: int,
        num_latents=3,
        num_tasks=3,
        kernel: str = "rbf",
    ):
        super().__init__()
        num_dim = 5  # TODO
        grid_bounds = (10, 10)  # TODO
        self.feature_extractor = TimeSeriesFeatureExtractor()
        self.gp_layer = GaussianProcessLayer(num_dim=num_dim, grid_bounds=grid_bounds)


class DKLGP(BaseLightningModule):
    def __init__(
        self,
        model,
        num_dim: int,
        train_dataset_length: int,
        learning_rate: float = 0.1,
        grid_bounds=(-10.0, 10.0),
    ):
        super(DKLGP, self).__init__()
        self.grid_bounds = grid_bounds
        self.num_dim = num_dim

        self.model = model

        # This module will scale the NN features so that they're nice values
        self.scale_to_bounds = gpytorch.utils.grid.ScaleToBounds(
            self.grid_bounds[0], self.grid_bounds[1]
        )

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=model.num_tasks)

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, model.gp_layer, num_data=train_dataset_length
        )

        self.learning_rate = learning_rate

    def model_forward(self, look_back_window: torch.Tensor):
        T = self.trainer.datamodule.prediction_window
        look_back_window = rearrange(
            look_back_window, "B T C -> B (T C)"
        )  # GPytorch assumes flattened channels
        preds = self.model(look_back_window)
        preds = rearrange(preds.mean, "B (T C) -> B T C", T=T)
        return preds

    def _shared_step(self, look_back_window, prediction_window):
        _, _, C = prediction_window.shape
        look_back_window = rearrange(
            look_back_window, "B T C -> B (T C)"
        )  # GPytorch assumes flattened channels
        preds = self.model(look_back_window)
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")
        loss = -self.mll(preds, prediction_window)
        return loss

    def model_specific_train_step(self, look_back_window, prediction_window):
        loss = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        loss = self._shared_step(look_back_window, prediction_window)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[0.5 * self.trainer.max_epochs, 0.75 * self.trainer.max_epochs],
            gamma=0.1,
        )

        return optimizer
