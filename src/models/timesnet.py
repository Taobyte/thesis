# ---------------------------------------------------------------
# This file includes code adapted from the Time-Series-Library:
# https://github.com/thuml/Time-Series-Library
#
# Original license: MIT License
# Copyright (c) THUML
#
# If you use this code, please consider citing the original repo:
# https://github.com/thuml/Time-Series-Library
# ---------------------------------------------------------------

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from torch import Tensor
from typing import Any, Tuple

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer,
    epoch: int,
    lradj: str,
    learning_rate: float,
    train_epochs: int = 10,
):
    lr_adjust: dict[int, float] = {}
    # lr = learning_rate * (0.2 ** (epoch // 2))
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "type3":
        lr_adjust = {
            epoch: learning_rate
            if epoch < 3
            else learning_rate * (0.9 ** ((epoch - 3) // 1))
        }
    elif lradj == "cosine":
        lr_adjust = {
            epoch: learning_rate / 2 * (1 + math.cos(epoch / train_epochs * math.pi))
        }
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        # print("Updating learning rate to {}".format(lr))
        return lr
    return optimizer.param_groups[0]["lr"]


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.tokenConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
            bias=False,
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="fixed", freq="h"):
        super(TemporalEmbedding, self).__init__()

        minute_size = 4
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == "fixed" else nn.Embedding
        if freq == "t":
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()
        minute_x = (
            self.minute_embed(x[:, :, 4]) if hasattr(self, "minute_embed") else 0.0
        )
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type="timeF", freq="h"):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {"h": 4, "t": 5, "s": 6, "m": 1, "a": 1, "w": 2, "d": 3, "b": 3}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = (
            TemporalEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
            if embed_type != "timeF"
            else TimeFeatureEmbedding(d_model=d_model, embed_type=embed_type, freq=freq)
        )
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            x = (
                self.value_embedding(x)
                + self.temporal_embedding(x_mark)
                + self.position_embedding(x)
            )
        return self.dropout(x)


class Inception_Block_V1(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(Inception_Block_V1, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i)
            )
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


def FFT_for_Period(x, k=2):
    # [B, T, C]

    xf = torch.fft.rfft(x, dim=1)
    # find period by amplitudes
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(
        frequency_list, min(k, len(frequency_list) - 1)
    )  # added min for k to not select too many indices if we only have l_b_w = 3
    top_list = top_list.detach().cpu().numpy()

    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        top_k: int,
        d_model: int,
        d_ff: int,
        num_kernels: int,
    ):
        super(TimesBlock, self).__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels),
        )

    def forward(self, x: Tensor):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)

        res: list[Tensor] = []
        for i in range(
            min(self.k, len(period_list))
        ):  # we need to loop over the length used in FFT_for_Period
            period = period_list[i]

            # padding
            if (self.look_back_window + self.prediction_window) % period != 0:
                length = (
                    ((self.look_back_window + self.prediction_window) // period) + 1
                ) * period
                padding = torch.zeros(
                    [
                        x.shape[0],
                        (length - (self.look_back_window + self.prediction_window)),
                        x.shape[2],
                    ]
                ).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = self.look_back_window + self.prediction_window
                out = x
            # reshape
            out = (
                out.reshape(B, length // period, period, N)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, : (self.look_back_window + self.prediction_window), :])
        # pdb.set_trace()
        res = torch.stack(res, dim=-1)
        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)
        # residual connection
        res = res + x
        return res


class Model(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """

    def __init__(
        self,
        look_back_window: int = 128,
        prediction_window: int = 64,
        c_out: int = 1,
        top_k: int = 5,
        d_model: int = 32,
        d_ff: int = 32,
        num_kernels: int = 6,
        e_layers: int = 2,
        enc_in: int = 1,
        embed_type: str = "fixed",
        freq: str = "h",
        dropout: float = 0.0,
        use_norm: bool = True,
    ):
        super(Model, self).__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.model = nn.ModuleList(
            [
                TimesBlock(
                    look_back_window,
                    prediction_window,
                    top_k,
                    d_model,
                    d_ff,
                    num_kernels,
                )
                for _ in range(e_layers)
            ]
        )
        self.enc_embedding = DataEmbedding(
            enc_in,
            d_model,
            embed_type,
            freq,
            dropout,
        )
        self.layer = e_layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.predict_linear = nn.Linear(
            self.look_back_window, self.prediction_window + self.look_back_window
        )
        self.projection = nn.Linear(d_model, c_out, bias=True)

        self.use_norm = use_norm

    def forecast(self, x_enc: torch.Tensor, x_mark_enc: torch.Tensor) -> torch.Tensor:
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        # TimesNet
        for i in range(self.layer):
            enc_out = self.layer_norm(self.model[i](enc_out))
        # project back

        dec_out = self.projection(enc_out)

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (
                stdev[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.prediction_window + self.look_back_window, 1)
            )

            dec_out = dec_out + (
                means[:, 0, :]
                .unsqueeze(1)
                .repeat(1, self.prediction_window + self.look_back_window, 1)
            )
        return dec_out

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.prediction_window :, :]  # [B, L, D]


class TimesNet(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss_fn: str = "MSE",
        lradj: str = "type1",
        beta_1: float = 0.9,
        beta_2: float = 0.999,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.lradj = lradj
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.model = model
        self.criterion = get_loss_fn(loss_fn)
        self.learning_rate = learning_rate

    def _generate_time_tensor(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        return torch.zeros((B, T, 5), device=x.device).float()

    def model_forward(self, look_back_window: Tensor):
        time = self._generate_time_tensor(look_back_window)
        preds = self.model(look_back_window, time)
        return preds

    def _shared_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor]:
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]  # remove activity channels

        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        mae_criterion = torch.nn.L1Loss()
        mae_loss = mae_criterion(preds, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        val_loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            val_loss = mae_loss
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True)
        return val_loss

    def on_train_epoch_end(self):
        current_lr = adjust_learning_rate(
            optimizer=self.trainer.optimizers[0],
            epoch=self.current_epoch + 1,
            lradj=self.lradj,
            learning_rate=self.learning_rate,
            train_epochs=self.trainer.max_epochs,
        )
        self.log("current_lr", current_lr, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            betas=(self.beta_1, self.beta_2),
        )

        return optimizer
