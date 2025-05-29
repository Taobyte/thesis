import torch
import gpytorch

from einops import rearrange
from gpytorch.models import ApproximateGP
from gpytorch.variational import (
    LMCVariationalStrategy,
)
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from typing import Tuple

from src.models.utils import BaseLightningModule

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
        use_linear_trend: bool = False,
        periodic_type: str = "",
    ):
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.train_dataset_length = train_dataset_length
        # we first have to rearrange the inducing points to have only two dimensions
        inducing_points = rearrange(inducing_points, "B T C -> B (T C)")
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

    def forward(self, x):
        # pdb.set_trace()
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class GaussianProcess(BaseLightningModule):
    def __init__(self, model, learning_rate: int = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.likelihood = MultitaskGaussianLikelihood(num_tasks=model.num_tasks)

        self.mll = VariationalELBO(
            self.likelihood, self.model, num_data=model.train_dataset_length
        )
        self.mae_loss = torch.nn.L1Loss()

    def model_forward(self, look_back_window: torch.Tensor):
        T = self.trainer.datamodule.prediction_window
        look_back_window = rearrange(
            look_back_window, "B T C -> B (T C)"
        )  # GPytorch assumes flattened channels
        preds = self.model(look_back_window)
        mean = rearrange(preds.mean, "B (T C) -> B T C", T=T)
        std = rearrange(preds.stddev, "B (T C) -> B T C", T=T)
        return mean, std

    def _shared_step(
        self, look_back_window, prediction_window
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _, _, C = prediction_window.shape
        look_back_window = rearrange(
            look_back_window, "B T C -> B (T C)"
        )  # GPytorch assumes flattened channels
        preds = self.model(look_back_window)
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")
        loss = -self.mll(preds, prediction_window)
        mae_loss = self.mae_loss(preds.mean, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(self, look_back_window, prediction_window):
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
