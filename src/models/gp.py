import math
import torch
import gpytorch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from gpytorch.models import ApproximateGP
from gpytorch.means import ConstantMean, LinearMean
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.variational import (
    VariationalStrategy,
    CholeskyVariationalDistribution,
    LMCVariationalStrategy,
)
from gpytorch.distributions import MultivariateNormal
from gpytorch.models.deep_gps import DeepGPLayer, DeepGP
from gpytorch.mlls import DeepApproximateMLL, VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import MultiStepLR

from src.models.utils import BaseLightningModule

# -----------------------------------------------------------------------------
# Deep Gaussian Process
# -----------------------------------------------------------------------------


class DGPHiddenLayer(DeepGPLayer):
    def __init__(self, input_dims, output_dims, num_inducing=128, linear_mean=True):
        inducing_points = torch.randn(output_dims, num_inducing, input_dims)
        batch_shape = torch.Size([output_dims])

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=num_inducing, batch_shape=batch_shape
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy, input_dims, output_dims)
        self.mean_module = ConstantMean() if linear_mean else LinearMean(input_dims)
        self.covar_module = ScaleKernel(
            MaternKernel(nu=2.5, batch_shape=batch_shape, ard_num_dims=input_dims),
            batch_shape=batch_shape,
            ard_num_dims=None,
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)


class MultitaskDeepGP(DeepGP):
    def __init__(self, train_x_shape, num_tasks: int = 3):
        hidden_layer = DGPHiddenLayer(
            input_dims=train_x_shape[-1],
            #  output_dims=num_hidden_dgp_dims, TODO
            linear_mean=True,
        )
        last_layer = DGPHiddenLayer(
            input_dims=hidden_layer.output_dims,
            output_dims=num_tasks,
            linear_mean=False,
        )

        super().__init__()

        self.hidden_layer = hidden_layer
        self.last_layer = last_layer

        # We're going to use a ultitask likelihood instead of the standard GaussianLikelihood
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

    def forward(self, inputs):
        hidden_rep1 = self.hidden_layer(inputs)
        output = self.last_layer(hidden_rep1)
        return output

    """
    def predict(self, test_x):
        with torch.no_grad():
            # The output of the model is a multitask MVN, where both the data points
            # and the tasks are jointly distributed
            # To compute the marginal predictive NLL of each data point,
            # we will call `to_data_independent_dist`,
            # which removes the data cross-covariance terms from the distribution.
            preds = model.likelihood(model(test_x)).to_data_independent_dist()

        return preds.mean.mean(0), preds.variance.mean(0)
    """


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
        optimizer = SGD(
            [
                {
                    "params": self.model.feature_extractor.parameters(),
                    "weight_decay": 1e-4,
                },
                {
                    "params": self.model.gp_layer.hyperparameters(),
                    "lr": self.learning_rate * 0.01,
                },
                {"params": self.model.gp_layer.variational_parameters()},
                {"params": self.likelihood.parameters()},
            ],
            lr=self.learning_rate,
            momentum=0.9,
            nesterov=True,
            weight_decay=0,
        )
        scheduler = MultiStepLR(
            optimizer,
            milestones=[0.5 * self.trainer.max_epochs, 0.75 * self.trainer.max_epochs],
            gamma=0.1,
        )

        return optimizer


# -----------------------------------------------------------------------------
# Standard Gaussian Process
# -----------------------------------------------------------------------------


class GPModel(ApproximateGP):
    def __init__(
        self,
        inducing_points,
        train_dataset_length: int,
        num_latents=3,
        num_tasks=3,
        kernel: str = "rbf",
    ):
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.train_dataset_length = train_dataset_length

        inducing_points = inducing_points.unsqueeze(0).expand(num_latents, -1, -1)

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

        if kernel == "periodic":
            base_kernel = gpytorch.kernels.PeriodicKernel(
                batch_shape=torch.Size([num_latents]),
                ard_num_dims=inducing_points.shape[-1],
            )
        elif kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel(
                batch_shape=torch.Size([num_latents])
            )
        elif kernel == "matern":
            base_kernel = gpytorch.kernels.MaternKernel(
                nu=2.5, batch_shape=torch.Size([num_latents])
            )
        elif kernel == "sm":
            base_kernel = gpytorch.kernels.SpectralMixtureKernel(
                num_mixtures=12,
                batch_shape=torch.Size([num_latents]),
                ard_num_dims=inducing_points.shape[-1],
            )

        # hardcode kernel
        base_kernel = (
            gpytorch.kernels.LinearKernel()
            + gpytorch.kernels.MaternKernel(
                nu=2.5, batch_shape=torch.Size([num_latents])
            )
            + gpytorch.kernels.PeriodicKernel(
                batch_shape=torch.Size([num_latents]),
                ard_num_dims=inducing_points.shape[-1],
            )
        )

        if kernel in ["sm"]:
            self.covar_module = base_kernel  # spectralMixture kernel should not be combined with scalekernel
        else:
            self.covar_module = gpytorch.kernels.ScaleKernel(
                base_kernel, batch_shape=torch.Size([num_latents])
            )

    def forward(self, x):
        # pdb.set_trace()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess(BaseLightningModule):
    def __init__(self, model, learning_rate: int = 0.001):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=model.num_tasks)

        self.mll = VariationalELBO(
            self.likelihood, self.model, num_data=model.train_dataset_length
        )

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
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
