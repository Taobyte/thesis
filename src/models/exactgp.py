import torch
import numpy as np
import gpytorch

from einops import rearrange
from gpytorch.likelihoods import MultitaskGaussianLikelihood
from typing import Tuple

from src.models.utils import BaseLightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        lbw_train_dataset: np.ndarray,
        pw_train_dataset: np.ndarray,
        lbw_val_dataset: np.ndarray,
        pw_val_dataset: np.ndarray,
        num_tasks=3,
        kernel: str = "rbf",
        use_linear_trend: bool = False,
        periodic_type: str = "",
    ):
        lbw_train_dataset = torch.tensor(lbw_train_dataset, device=device)
        pw_train_dataset = torch.tensor(pw_train_dataset, device=device)
        lbw_train_dataset = rearrange(lbw_train_dataset, "B T C -> B (T C)")
        pw_train_dataset = rearrange(pw_train_dataset, "B T C -> B (T C)")
        likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)

        super(ExactGPModel, self).__init__(
            lbw_train_dataset, pw_train_dataset, likelihood
        )

        self.likelihood = likelihood

        if kernel == "rbf":
            kernel = gpytorch.kernels.RBFKernel()
        elif kernel == "matern":
            kernel = gpytorch.kernels.MaternKernel(
                nu=2.5,
            )
        elif kernel == "rq":
            kernel = gpytorch.kernels.RQKernel()
        else:
            raise NotImplementedError()

        #         elif kernel == "sm":
        #             kernel = gpytorch.kernels.SpectralMixtureKernel(
        #                 num_mixtures=12,
        #                 batch_shape=torch.Size([num_latents]),
        #                 ard_num_dims=inducing_points.shape[-1],
        #             )
        if periodic_type == "multiplicative":
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            kernel *= periodic_kernel
        elif periodic_type == "additive":
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            kernel += periodic_kernel

        if use_linear_trend:
            kernel += gpytorch.kernels.LinearKernel()

        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        self.covar_module = gpytorch.kernels.MultitaskKernel(
            kernel, num_tasks=num_tasks, rank=1
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGaussianProcess(BaseLightningModule):
    def __init__(self, model, learning_rate: int = 0.001, **kwargs):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.likelihood = model.likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)
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
