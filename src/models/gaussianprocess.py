import pandas as pd
import numpy as np
import torch
import gpytorch

from src.models.utils import BaseLightningModule
from gpytorch.kernels import SpectralMixtureKernel, RBFKernel, ScaleKernel, MaternKernel
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy


class GPModel(ApproximateGP):
    def __init__(self, inducing_points):
        variational_distribution = CholeskyVariationalDistribution(
            inducing_points.size(0)
        )
        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class SpectralMixtureGPModel(gpytorch.models.ExactGP):
    def __init__(self, likelihood):
        super().__init__(likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        covar = SpectralMixtureKernel(num_mixtures=12)
        self.covar_module = (
            ScaleKernel(covar) + RBFKernel()
        )  # +ScaleKernel(MaternKernel)
        # self.covar_module.initialize_from_data(train_x, train_y)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess(BaseLightningModule):
    def __init__(self, learning_rate: int = 0.001):
        self.model = SpectralMixtureGPModel()
        self.learning_rate = learning_rate
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood()

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, self.model, num_data=1
        )

    def model_specific_train_step(self, look_back_window, prediction_window):
        return super().model_specific_train_step(look_back_window, prediction_window)

    def model_specific_val_step(self, look_back_window, prediction_window):
        return super().model_specific_val_step(look_back_window, prediction_window)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
