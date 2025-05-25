import numpy as np
import torch
import pyro
import pyro.distributions as dist
import torch.nn as nn

from pyro.nn import PyroModule, PyroSample
from pyro.infer import SVI, Trace_ELBO, Predictive
from pyro.infer.autoguide import AutoDiagonalNormal, AutoMultivariateNormal
from einops import rearrange

from src.models.utils import BaseLightningModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model(PyroModule):
    def __init__(
        self,
        look_back_window: int = 5,
        prediction_window: int = 3,
        base_channel_dim: int = 1,
        input_channels: int = 1,
        hid_dim=10,
        n_hid_layers=5,
        prior_scale=5.0,
        activation: str = "tanh",
        heteroscedastic: bool = False,
        output_noise_sigma: float = 0.01,
    ):
        super().__init__()

        self.base_channel_dim = base_channel_dim

        if activation == "relu":
            self.activation = torch.nn.ReLU()
        elif activation == "tanh":
            self.activation = torch.nn.Tanh()
        elif activation == "none":
            self.activation = torch.nn.Identity()
        else:
            raise NotImplementedError()

        self.heteroscedastic = heteroscedastic
        self.output_noise_sigma = output_noise_sigma

        in_dim = look_back_window * (input_channels)
        out_dim = prediction_window * base_channel_dim

        # Define the layer sizes and the PyroModule layer list
        self.layer_sizes = [in_dim] + n_hid_layers * [hid_dim] + [out_dim]
        layer_list = [
            PyroModule[nn.Linear](self.layer_sizes[idx - 1], self.layer_sizes[idx])
            for idx in range(1, len(self.layer_sizes))
        ]
        self.layers = PyroModule[torch.nn.ModuleList](layer_list)

        for layer_idx, layer in enumerate(self.layers):
            layer.weight = PyroSample(
                dist.Normal(
                    torch.tensor(0.0, device=device),
                    prior_scale * np.sqrt(2 / self.layer_sizes[layer_idx]),
                )
                .expand([self.layer_sizes[layer_idx + 1], self.layer_sizes[layer_idx]])
                .to_event(2)
            )
            layer.bias = PyroSample(
                dist.Normal(torch.tensor(0.0, device=device), prior_scale)
                .expand([self.layer_sizes[layer_idx + 1]])
                .to_event(1)
            )

    def forward(self, x, y=None):
        x = self.activation(self.layers[0](x))  # input --> hidden
        for layer in self.layers[1:-1]:
            x = self.activation(layer(x))  # hidden --> hidden
        mu = self.layers[-1](x)  # hidden --> output
        """
        sigma = pyro.sample(
            "sigma", dist.Gamma(torch.tensor(0.5, device=device), 1)
        )  # infer the response noise
        """
        sigma = torch.tensor(self.output_noise_sigma, device=device)
        # this part computes the log likelihood during training
        with pyro.plate("data", x.shape[0]):
            obs = pyro.sample("obs", dist.Normal(mu, sigma * sigma).to_event(1), obs=y)
        return mu


class BayesianNeuralNetwork(BaseLightningModule):
    def __init__(
        self,
        model: PyroModule,
        learning_rate: int = 0.01,
        num_samples: int = 500,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.automatic_optimization = False  # we optimize the BNN using pyro
        self.model = model

        mean_field_guide = AutoMultivariateNormal(model)
        optimizer = pyro.optim.Adam({"lr": learning_rate})

        self.svi = SVI(model, mean_field_guide, optimizer, loss=Trace_ELBO())
        pyro.clear_param_store()

        self.predictive = Predictive(
            self.model, guide=mean_field_guide, num_samples=num_samples
        )

    def model_forward(self, look_back_window):
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        preds = self.predictive(look_back_window)  # (n_samples, B, T*C)
        mean_preds = preds["obs"].mean(dim=0)  # (B, T*C)

        reshaped_mean_preds = rearrange(
            mean_preds, "B (T C) -> B T C", C=self.model.base_channel_dim
        )
        return reshaped_mean_preds

    def model_specific_train_step(self, look_back_window, prediction_window):
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")

        loss = self.svi.step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return None

    def model_specific_val_step(self, look_back_window, prediction_window):
        look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        prediction_window = rearrange(prediction_window, "B T C -> B (T C)")
        if self.tune:
            preds = self.model_forward(look_back_window)
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, prediction_window)
        else:
            loss = self.svi.evaluate_loss(look_back_window, prediction_window)
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return None

    def configure_optimizers(self):
        return None
