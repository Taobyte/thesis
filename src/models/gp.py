import torch
import gpytorch


from einops import rearrange
from gpytorch.models import ApproximateGP

from src.models.utils import BaseLightningModule


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
        variational_strategy = gpytorch.variational.LMCVariationalStrategy(
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
        self.likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(
            num_tasks=model.num_tasks
        )

        self.mll = gpytorch.mlls.VariationalELBO(
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
