import torch
import gpytorch

from torch import Tensor
from einops import rearrange
from gpytorch.likelihoods import MultitaskGaussianLikelihood

from typing import Tuple, Any

from src.models.utils import BaseLightningModule


class MultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(
        self,
        lbw_train_dataset,
        pw_train_dataset,
        lbw_val_dataset,
        pw_val_dataset,
        num_tasks: int,
        kernel: str = "rbf",
        periodic_type: str = "",
        use_linear_trend: bool = True,
    ):
        lbw_train_dataset = rearrange(lbw_train_dataset, "B T C -> B (T C)")
        pw_train_dataset = rearrange(pw_train_dataset, "B T C -> B (T C)")
        lbw_val_dataset = rearrange(lbw_val_dataset, "B T C -> B (T C)")
        pw_val_dataset = rearrange(pw_val_dataset, "B T C -> B (T C)")
        likelihood = MultitaskGaussianLikelihood(num_tasks=num_tasks)
        super(MultitaskGPModel, self).__init__(
            torch.from_numpy(lbw_train_dataset),
            torch.from_numpy(pw_train_dataset),
            likelihood,
        )

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lbw_train_dataset = torch.from_numpy(lbw_train_dataset).to(device)
        self.pw_train_dataset = torch.from_numpy(pw_train_dataset).to(device)
        self.lbw_val_dataset = torch.from_numpy(lbw_val_dataset).to(device)
        self.pw_val_dataset = torch.from_numpy(pw_val_dataset).to(device)

        self.num_tasks = num_tasks
        self.mean_module = gpytorch.means.MultitaskMean(
            gpytorch.means.ConstantMean(), num_tasks=num_tasks
        )

        if kernel == "rbf":
            base_kernel = gpytorch.kernels.RBFKernel()
        elif kernel == "matern":
            base_kernel = gpytorch.kernels.MaternKernel()
        elif kernel == "rq":
            base_kernel = gpytorch.kernels.RQKernel()
        else:
            raise NotImplementedError()

        if periodic_type == "multiplicative":
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            base_kernel *= periodic_kernel
        elif periodic_type == "additive":
            periodic_kernel = gpytorch.kernels.PeriodicKernel()
            base_kernel += periodic_kernel

        if use_linear_trend:
            base_kernel += gpytorch.kernels.LinearKernel()

        if kernel in ["sm"]:
            self.covar_module = base_kernel
        else:
            self.covar_module = gpytorch.kernels.MultitaskKernel(
                gpytorch.kernels.ScaleKernel(base_kernel), num_tasks=num_tasks, rank=1
            )

    def forward(self, x: Tensor):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


class ExactGP(BaseLightningModule):
    def __init__(
        self,
        model: MultitaskGPModel,
        learning_rate: float = 0.001,
        use_norm: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.learning_rate = learning_rate

        self.model = model
        self.likelihood = model.likelihood
        self.mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.likelihood, model)

        self.mae_loss = torch.nn.L1Loss()
        self.use_norm = use_norm

        self.automatic_optimization = False

    def model_forward(self, look_back_window: torch.Tensor) -> Tuple[Tensor, Tensor]:
        if self.use_norm:
            means = look_back_window.mean(1, keepdim=True).detach()
            look_back_window = look_back_window - means
            stdev = torch.sqrt(
                torch.var(look_back_window, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            look_back_window /= stdev
        T = self.trainer.datamodule.prediction_window
        look_back_window = rearrange(
            look_back_window, "B T C -> B (T C)"
        )  # GPytorch assumes flattened channels
        preds = self.model(look_back_window)

        mean = rearrange(preds.mean, "B (T C) -> B T C", T=T)
        std = rearrange(preds.stddev, "B (T C) -> B T C", T=T)

        if self.use_norm:
            mean = mean * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )

            mean = mean + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)
            )

            std = std * stdev[:, 0, :].unsqueeze(1).repeat(1, self.prediction_window, 1)

        return mean, std

    def on_train_epoch_start(self):
        device = self.device
        self.model.train()
        self.likelihood.train()

        optimizer = self.optimizers()
        optimizer.zero_grad()
        output = self.model(self.model.lbw_train_dataset)
        loss = -self.mll(output, self.model.pw_train_dataset)
        self.manual_backward(loss)
        optimizer.step()

    def on_validation_epoch_start(self):
        val_preds = self.model(self.model.lbw_val_dataset)
        val_loss = -self.mll(val_preds, self.model.pw_val_dataset)
        self.log("val_loss", val_loss)

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        return None

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ):
        return None

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer
