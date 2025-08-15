import math
import numpy as np
import torch as t
import torch.nn as nn

from functools import partial

from torch.nn.utils import weight_norm
from typing import Tuple, List, Union, Any

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


## We use the implementation of TCN from https://github.com/locuslab/TCN


def init_weights(module, initialization):
    if type(module) == t.nn.Linear:
        if initialization == "orthogonal":
            t.nn.init.orthogonal_(module.weight)
        elif initialization == "he_uniform":
            t.nn.init.kaiming_uniform_(module.weight)
        elif initialization == "he_normal":
            t.nn.init.kaiming_normal_(module.weight)
        elif initialization == "glorot_uniform":
            t.nn.init.xavier_uniform_(module.weight)
        elif initialization == "glorot_normal":
            t.nn.init.xavier_normal_(module.weight)
        elif initialization == "lecun_normal":
            pass
        else:
            assert 1 < 0, f"Initialization {initialization} not found"


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(
        self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2
    ):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                n_inputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv1d(
                n_outputs,
                n_outputs,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
            )
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )
        self.downsample = (
            nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        )
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            ]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class _StaticFeaturesEncoder(nn.Module):
    def __init__(self, in_features, out_features):
        super(_StaticFeaturesEncoder, self).__init__()
        layers = [
            nn.Dropout(p=0.5),
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.ReLU(),
        ]
        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        return x


class NBeatsBlock(nn.Module):
    """
    N-BEATS block which takes a basis function as an argument.
    """

    def __init__(
        self,
        x_t_n_inputs: int,
        x_s_n_inputs: int,
        x_s_n_hidden: int,
        theta_n_dim: int,
        basis: nn.Module,
        n_layers: int,
        theta_n_hidden: list[int],
        batch_normalization: bool,
        dropout_prob: float,
        activation: str,
    ):
        """ """
        super().__init__()

        if x_s_n_inputs == 0:
            x_s_n_hidden = 0
        theta_n_hidden = [x_t_n_inputs + x_s_n_hidden] + theta_n_hidden

        self.x_s_n_inputs = x_s_n_inputs
        self.x_s_n_hidden = x_s_n_hidden
        self.batch_normalization = batch_normalization
        self.dropout_prob = dropout_prob
        self.activations = {
            "relu": nn.ReLU(),
            "softplus": nn.Softplus(),
            "tanh": nn.Tanh(),
            "selu": nn.SELU(),
            "lrelu": nn.LeakyReLU(),
            "prelu": nn.PReLU(),
            "sigmoid": nn.Sigmoid(),
        }

        hidden_layers = []
        for i in range(n_layers):
            # Batch norm after activation
            hidden_layers.append(
                nn.Linear(
                    in_features=theta_n_hidden[i], out_features=theta_n_hidden[i + 1]
                )
            )
            hidden_layers.append(self.activations[activation])

            if self.batch_normalization:
                hidden_layers.append(nn.BatchNorm1d(num_features=theta_n_hidden[i + 1]))

            if self.dropout_prob > 0:
                hidden_layers.append(nn.Dropout(p=self.dropout_prob))

        output_layer = [
            nn.Linear(in_features=theta_n_hidden[-1], out_features=theta_n_dim)
        ]
        layers = hidden_layers + output_layer

        # x_s_n_inputs is computed with data, x_s_n_hidden is provided by user, if 0 no statics are used
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            self.static_encoder = _StaticFeaturesEncoder(
                in_features=x_s_n_inputs, out_features=x_s_n_hidden
            )
        self.layers = nn.Sequential(*layers)
        self.basis = basis

    def forward(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
    ) -> Tuple[t.Tensor, t.Tensor]:
        # Static exogenous
        if (self.x_s_n_inputs > 0) and (self.x_s_n_hidden > 0):
            x_s = self.static_encoder(x_s)
            insample_y = t.cat((insample_y, x_s), 1)

        # Compute local projection weights and projection
        theta = self.layers(insample_y)
        backcast, forecast = self.basis(theta, insample_x_t, outsample_x_t)

        return backcast, forecast


class NBeats(nn.Module):
    """
    N-Beats Model.
    """

    def __init__(self, blocks: nn.ModuleList):
        super().__init__()
        self.blocks = blocks

    def forward(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        insample_mask: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
        return_decomposition: bool = False,
    ) -> Union[t.Tensor, Tuple[t.Tensor, t.Tensor]]:
        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        block_forecasts: List[t.Tensor] = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(
                insample_y=residuals,
                insample_x_t=insample_x_t,
                outsample_x_t=outsample_x_t,
                x_s=x_s,
            )
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            block_forecasts.append(block_forecast)

        # (n_batch, n_blocks, n_time)
        block_forecasts = t.stack(block_forecasts)
        block_forecasts = block_forecasts.permute(1, 0, 2)

        if return_decomposition:
            return forecast, block_forecasts
        else:
            return forecast

    def decomposed_prediction(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        insample_mask: t.Tensor,
        outsample_x_t: t.Tensor,
    ):
        residuals = insample_y.flip(dims=(-1,))
        insample_x_t = insample_x_t.flip(dims=(-1,))
        insample_mask = insample_mask.flip(dims=(-1,))

        forecast = insample_y[:, -1:]  # Level with Naive1
        forecast_components = []
        for i, block in enumerate(self.blocks):
            backcast, block_forecast = block(residuals, insample_x_t, outsample_x_t)
            residuals = (residuals - backcast) * insample_mask
            forecast = forecast + block_forecast
            forecast_components.append(block_forecast)
        return forecast, forecast_components


class IdentityBasis(nn.Module):
    def __init__(self, backcast_size: int, forecast_size: int):
        super().__init__()
        self.forecast_size = forecast_size
        self.backcast_size = backcast_size

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        backcast = theta[:, : self.backcast_size]
        forecast = theta[:, -self.forecast_size :]
        return backcast, forecast


class TrendBasis(nn.Module):
    def __init__(
        self, degree_of_polynomial: int, backcast_size: int, forecast_size: int
    ):
        super().__init__()
        polynomial_size = degree_of_polynomial + 1
        self.backcast_basis = nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(backcast_size, dtype=np.float32) / backcast_size,
                            i,
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )
        self.forecast_basis = nn.Parameter(
            t.tensor(
                np.concatenate(
                    [
                        np.power(
                            np.arange(forecast_size, dtype=np.float32) / forecast_size,
                            i,
                        )[None, :]
                        for i in range(polynomial_size)
                    ]
                ),
                dtype=t.float32,
            ),
            requires_grad=False,
        )

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        cut_point = self.forecast_basis.shape[0]
        backcast = t.einsum("bp,pt->bt", theta[:, cut_point:], self.backcast_basis)
        forecast = t.einsum("bp,pt->bt", theta[:, :cut_point], self.forecast_basis)
        return backcast, forecast


class ExogenousBasisInterpretable(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis = insample_x_t
        forecast_basis = outsample_x_t

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum("bp,bpt->bt", theta[:, cut_point:], backcast_basis)
        forecast = t.einsum("bp,bpt->bt", theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class ExogenousBasisWavenet(nn.Module):
    def __init__(
        self, out_features, in_features, num_levels=4, kernel_size=3, dropout_prob=0
    ):
        super().__init__()
        # Shape of (1, in_features, 1) to broadcast over b and t
        self.weight = nn.Parameter(t.Tensor(1, in_features, 1), requires_grad=True)
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(0.5))

        padding = (kernel_size - 1) * (2**0)
        input_layer = [
            nn.Conv1d(
                in_channels=in_features,
                out_channels=out_features,
                kernel_size=kernel_size,
                padding=padding,
                dilation=2**0,
            ),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout_prob),
        ]
        conv_layers = []
        for i in range(1, num_levels):
            dilation = 2**i
            padding = (kernel_size - 1) * dilation
            conv_layers.append(
                nn.Conv1d(
                    in_channels=out_features,
                    out_channels=out_features,
                    padding=padding,
                    kernel_size=kernel_size,
                    dilation=dilation,
                )
            )
            conv_layers.append(Chomp1d(padding))
            conv_layers.append(nn.ReLU())
        conv_layers = input_layer + conv_layers

        self.wavenet = nn.Sequential(*conv_layers)

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = (
            x_t * self.weight
        )  # Element-wise multiplication, broadcasted on b and t. Weights used in L1 regularization
        x_t = self.wavenet(x_t)[:]

        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum("bp,bpt->bt", theta[:, cut_point:], backcast_basis)
        forecast = t.einsum("bp,bpt->bt", theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class ExogenousBasisTCN(nn.Module):
    def __init__(
        self, out_features, in_features, num_levels=4, kernel_size=2, dropout_prob=0
    ):
        super().__init__()
        n_channels = num_levels * [out_features]
        self.tcn = TemporalConvNet(
            num_inputs=in_features,
            num_channels=n_channels,
            kernel_size=kernel_size,
            dropout=dropout_prob,
        )

    def transform(self, insample_x_t, outsample_x_t):
        input_size = insample_x_t.shape[2]

        x_t = t.cat([insample_x_t, outsample_x_t], dim=2)

        x_t = self.tcn(x_t)[:]
        backcast_basis = x_t[:, :, :input_size]
        forecast_basis = x_t[:, :, input_size:]

        return backcast_basis, forecast_basis

    def forward(
        self, theta: t.Tensor, insample_x_t: t.Tensor, outsample_x_t: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        backcast_basis, forecast_basis = self.transform(insample_x_t, outsample_x_t)

        cut_point = forecast_basis.shape[1]
        backcast = t.einsum("bp,bpt->bt", theta[:, cut_point:], backcast_basis)
        forecast = t.einsum("bp,bpt->bt", theta[:, :cut_point], forecast_basis)
        return backcast, forecast


class Model(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        shared_weights: bool,
        activation: str,
        initialization: str,
        stack_types: list[str],
        n_blocks: List[int],
        n_layers: List[int],
        n_hidden: int,
        n_polynomials: int,
        batch_normalization: bool,
        exogenous_n_channels: int,
        dropout_prob_theta: float,
        dropout_prob_exogenous: float,
        x_s_n_hidden: int,
        kernel_size: int = 2,
        n_x_s: int = 0,  # number of static exogenous variables
    ):
        super().__init__()
        self.input_size = input_size  # int(input_size_multiplier * output_size)
        self.output_size = output_size
        self.shared_weights = shared_weights
        self.activation = activation
        self.initialization = initialization
        self.stack_types = stack_types
        self.n_blocks = n_blocks
        self.n_layers = n_layers
        n_hidden_list: List[List[int]] = []
        for n in n_layers:
            n_hidden_list.append([n_hidden for _ in range(n)])
        self.n_hidden = n_hidden_list
        self.n_polynomials = n_polynomials
        self.exogenous_n_channels = exogenous_n_channels

        # Regularization and optimization parameters
        self.batch_normalization = batch_normalization
        self.dropout_prob_theta = dropout_prob_theta
        self.dropout_prob_exogenous = dropout_prob_exogenous
        self.x_s_n_hidden = x_s_n_hidden

        self.n_x_s = n_x_s
        self.n_x_t = 1  # hard coded, we either pass in zeros for endo_only or the activity info for endo_exo experiments

        self.kernel_size = kernel_size
        assert kernel_size <= input_size

        block_list = self.create_stack()
        block_module_list = nn.ModuleList(block_list)
        self.model = NBeats(block_module_list)

    def forward(
        self,
        insample_y: t.Tensor,
        insample_x_t: t.Tensor,
        insample_mask: t.Tensor,
        outsample_x_t: t.Tensor,
        x_s: t.Tensor,
        return_decomposition: bool = False,
    ):
        return self.model(
            insample_y,
            insample_x_t,
            insample_mask,
            outsample_x_t,
            x_s,
            return_decomposition=return_decomposition,
        )

    def create_stack(self) -> List[NBeatsBlock]:
        x_t_n_inputs = self.input_size

        # ------------------------ Model Definition ------------------------#
        block_list: List[NBeatsBlock] = []
        self.blocks_regularizer: List[int] = []
        for i in range(len(self.stack_types)):
            for block_id in range(self.n_blocks[i]):
                # Batch norm only on first block
                if (len(block_list) == 0) and (self.batch_normalization):
                    batch_normalization_block = True
                else:
                    batch_normalization_block = False

                # Dummy of regularizer in block. Override with 1 if exogenous_block
                self.blocks_regularizer += [0]

                # Shared weights
                if self.shared_weights and block_id > 0:
                    nbeats_block = block_list[-1]
                else:
                    # removed == seasonality branch!
                    if self.stack_types[i] == "trend":
                        nbeats_block = NBeatsBlock(
                            x_t_n_inputs=x_t_n_inputs,
                            x_s_n_inputs=self.n_x_s,
                            x_s_n_hidden=self.x_s_n_hidden,
                            theta_n_dim=2 * (self.n_polynomials + 1),
                            basis=TrendBasis(
                                degree_of_polynomial=self.n_polynomials,
                                backcast_size=self.input_size,
                                forecast_size=self.output_size,
                            ),
                            n_layers=self.n_layers[i],
                            theta_n_hidden=self.n_hidden[i],
                            batch_normalization=batch_normalization_block,
                            dropout_prob=self.dropout_prob_theta,
                            activation=self.activation,
                        )
                    elif self.stack_types[i] == "identity":
                        nbeats_block = NBeatsBlock(
                            x_t_n_inputs=x_t_n_inputs,
                            x_s_n_inputs=self.n_x_s,
                            x_s_n_hidden=self.x_s_n_hidden,
                            theta_n_dim=self.input_size + self.output_size,
                            basis=IdentityBasis(
                                backcast_size=self.input_size,
                                forecast_size=self.output_size,
                            ),
                            n_layers=self.n_layers[i],
                            theta_n_hidden=self.n_hidden[i],
                            batch_normalization=batch_normalization_block,
                            dropout_prob=self.dropout_prob_theta,
                            activation=self.activation,
                        )
                    elif self.stack_types[i] == "exogenous":
                        nbeats_block = NBeatsBlock(
                            x_t_n_inputs=x_t_n_inputs,
                            x_s_n_inputs=self.n_x_s,
                            x_s_n_hidden=self.x_s_n_hidden,
                            theta_n_dim=2 * self.n_x_t,
                            basis=ExogenousBasisInterpretable(),
                            n_layers=self.n_layers[i],
                            theta_n_hidden=self.n_hidden[i],
                            batch_normalization=batch_normalization_block,
                            dropout_prob=self.dropout_prob_theta,
                            activation=self.activation,
                        )
                    elif self.stack_types[i] == "exogenous_tcn":
                        nbeats_block = NBeatsBlock(
                            x_t_n_inputs=x_t_n_inputs,
                            x_s_n_inputs=self.n_x_s,
                            x_s_n_hidden=self.x_s_n_hidden,
                            theta_n_dim=2 * (self.exogenous_n_channels),
                            basis=ExogenousBasisTCN(
                                self.exogenous_n_channels,
                                self.n_x_t,
                                kernel_size=self.kernel_size,
                            ),
                            n_layers=self.n_layers[i],
                            theta_n_hidden=self.n_hidden[i],
                            batch_normalization=batch_normalization_block,
                            dropout_prob=self.dropout_prob_theta,
                            activation=self.activation,
                        )
                    elif self.stack_types[i] == "exogenous_wavenet":
                        nbeats_block = NBeatsBlock(
                            x_t_n_inputs=x_t_n_inputs,
                            x_s_n_inputs=self.n_x_s,
                            x_s_n_hidden=self.x_s_n_hidden,
                            theta_n_dim=2 * (self.exogenous_n_channels),
                            basis=ExogenousBasisWavenet(
                                self.exogenous_n_channels,
                                self.n_x_t,
                                kernel_size=self.kernel_size,
                            ),
                            n_layers=self.n_layers[i],
                            theta_n_hidden=self.n_hidden[i],
                            batch_normalization=batch_normalization_block,
                            dropout_prob=self.dropout_prob_theta,
                            activation=self.activation,
                        )
                        self.blocks_regularizer[-1] = 1
                    else:
                        assert 1 < 0, "Block type not found!"
                # Select type of evaluation and apply it to all layers of block
                init_function = partial(
                    init_weights, initialization=self.initialization
                )
                nbeats_block.layers.apply(init_function)
                block_list.append(nbeats_block)
        return block_list


class NBeatsX(BaseLightningModule):
    def __init__(
        self,
        model: Model,
        weight_decay: float,
        learning_rate: float = 0.001,
        loss_fn: str = "MSE",
        n_lr_decay_steps: int = 3,
        lr_decay: float = 0.5,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.learning_rate = learning_rate
        self.weigth_decay = weight_decay
        self.n_lr_decay_steps = n_lr_decay_steps
        self.lr_decay = lr_decay

        self.criterion = get_loss_fn(loss_fn=loss_fn)
        self.mae_criterion = nn.L1Loss()

    def model_forward(self, look_back_window: t.Tensor):
        B, T, C = look_back_window.shape
        heartrate = look_back_window[:, :, 0]  # (B, T)

        if self.use_dynamic_features:
            activity = look_back_window[:, :, 1].unsqueeze(1)  # (B, 1, T)
            # For future: you need actual future activity values, not zeros
        else:
            # If no exogenous features, you might need to handle this differently
            # Option 1: Use dummy features with proper shape
            activity = t.zeros((B, 1, T))  # (B, 1, T)
        outsample_x_t = t.zeros((B, 1, self.prediction_window))
        prediction = self.model.forward(
            insample_y=heartrate,  # (B, T)
            insample_x_t=activity,  # (B, 1, T)
            insample_mask=t.ones_like(heartrate),  # (B, T)
            outsample_x_t=outsample_x_t,  # (B, 1, H)
            x_s=None,
            return_decomposition=False,
        )

        return prediction.unsqueeze(-1)

    def _shared_step(
        self, look_back_window: t.Tensor, prediction_window: t.Tensor
    ) -> Tuple[t.Tensor, t.Tensor]:
        preds = self.model_forward(look_back_window)
        loss = self.criterion(preds, prediction_window)
        mae_loss = self.mae_criterion(preds, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: t.Tensor, prediction_window: t.Tensor
    ) -> t.Tensor:
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: t.Tensor, prediction_window: t.Tensor
    ) -> t.Tensor:
        val_loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        loss = val_loss
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = t.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weigth_decay,
        )
        lr_decay_steps = self.trainer.max_epochs // self.n_lr_decay_steps
        scheduler = t.optim.lr_scheduler.StepLR(
            optimizer, step_size=lr_decay_steps, gamma=self.lr_decay
        )

        scheduler_dict = {
            "scheduler": scheduler,
            "interval": "epoch",  # because OneCycleLR is stepped per batch
            "frequency": 1,
            "name": "StepLR",
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}
