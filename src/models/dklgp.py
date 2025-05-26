import torch
import gpytorch
import torch.nn as nn

from einops import rearrange
from gpytorch.variational import LMCVariationalStrategy
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from src.models.utils import BaseLightningModule

# -----------------------------------------------------------------------------
# SVDKL (Stochastic Variational Deep Kernel Learning)
# -----------------------------------------------------------------------------


class TimeSeriesFeatureExtractor(nn.Module):
    def __init__(
        self,
        input_channels: int,
        hidden_dim: int = 8,
        out_dim: int = 16,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv1d(input_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool1d(1),
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

    def forward(self, x):  # x: (B, T, C)
        x = x.permute(0, 2, 1)  # -> (B, C, T)
        x = self.net(x)  # -> (B, hidden_dim, 1)
        x = x.squeeze(-1)  # -> (B, hidden_dim)
        return self.fc(x)  # -> (B, out_dim)


class GaussianProcessLayer(gpytorch.models.ApproximateGP):
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

        inducing_points = inducing_points.unsqueeze(0).expand(num_latents, -1, -1)

        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(-2), batch_shape=torch.Size([num_latents])
        )

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
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean, covar)


class DKLGPModel(torch.nn.Module):
    def __init__(
        self,
        inducing_points,
        train_dataset_length: int,
        num_latents=3,
        num_tasks=3,
        kernel: str = "rbf",
        use_linear_trend: bool = False,
        periodic_type: str = "",
        look_back_channel_dim: int = 1,
        hidden_dim: int = 8,
        out_dim: int = 16,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.num_latents = num_latents
        self.num_tasks = num_tasks
        self.train_dataset_length = train_dataset_length

        self.feature_extractor = TimeSeriesFeatureExtractor(
            look_back_channel_dim, hidden_dim, out_dim, dropout
        )

        # transform the inducing points into the feature space
        inducing_points = rearrange(
            inducing_points, "I (T C) -> I T C", C=look_back_channel_dim
        )
        inducing_points = self.feature_extractor(inducing_points)
        self.gp_layer = GaussianProcessLayer(
            inducing_points,
            train_dataset_length,
            num_latents,
            num_tasks,
            kernel,
            use_linear_trend,
            periodic_type,
        )

    def forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        x = self.feature_extractor(look_back_window)
        x = self.gp_layer(x)
        return x


class DKLGP(BaseLightningModule):
    def __init__(
        self,
        model,
        learning_rate: float = 0.1,
        use_scheduler: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.learning_rate = learning_rate
        self.use_scheduler = use_scheduler

        self.likelihood = MultitaskGaussianLikelihood(num_tasks=model.num_tasks)

        self.mll = gpytorch.mlls.VariationalELBO(
            self.likelihood, model.gp_layer, num_data=model.train_dataset_length
        )
        self.mae_loss = torch.nn.L1Loss()

    def model_forward(self, look_back_window: torch.Tensor):
        T = self.trainer.datamodule.prediction_window
        preds = self.model(look_back_window)
        preds = rearrange(preds.mean, "B (T C) -> B T C", T=T)
        return preds

    def _shared_step(self, look_back_window, prediction_window):
        _, _, C = prediction_window.shape
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
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        if self.tune:
            loss = mae_loss
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        if self.use_scheduler:
            scheduler = torch.optim.MultiStepLR(
                optimizer,
                milestones=[
                    0.5 * self.trainer.max_epochs,
                    0.75 * self.trainer.max_epochs,
                ],
                gamma=0.1,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "epoch",
                    "frequency": 1,
                    "name": "MultiStepLR",
                },
            }

        return optimizer
