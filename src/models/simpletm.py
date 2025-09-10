# Code adapted from: https://github.com/vsingh-group/SimpleTM
# Zhou et al., "A Simple Baseline for Multivariate Time Series Forecasting", arXiv:2305.08897

import torch
import torch.nn as nn
import torch.nn.functional as F
import pywt

from math import sqrt
from torch import Tensor
from typing import Tuple, Any, Union, Dict

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


class EncoderLayer(nn.Module):
    def __init__(
        self,
        attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
        n_channels=866,
    ):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask, tau=tau, delta=delta)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = (
            nn.ModuleList(conv_layers) if conv_layers is not None else None
        )
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(
                zip(self.attn_layers, self.conv_layers)
            ):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class WaveletEmbedding(nn.Module):
    def __init__(
        self,
        d_channel=16,
        swt=True,
        requires_grad=False,
        wv="db2",
        m=2,
        kernel_size=None,
    ):
        super().__init__()

        self.swt = swt
        self.d_channel = d_channel
        self.m = m  # Number of decomposition levels of detailed coefficients

        if kernel_size is None:
            self.wavelet = pywt.Wavelet(wv)
            if self.swt:
                h0 = torch.tensor(self.wavelet.dec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.dec_hi[::-1], dtype=torch.float32)
            else:
                h0 = torch.tensor(self.wavelet.rec_lo[::-1], dtype=torch.float32)
                h1 = torch.tensor(self.wavelet.rec_hi[::-1], dtype=torch.float32)
            self.h0 = nn.Parameter(
                torch.tile(h0[None, None, :], [self.d_channel, 1, 1]),
                requires_grad=requires_grad,
            )
            self.h1 = nn.Parameter(
                torch.tile(h1[None, None, :], [self.d_channel, 1, 1]),
                requires_grad=requires_grad,
            )
            self.kernel_size = self.h0.shape[-1]
        else:
            self.kernel_size = kernel_size
            self.h0 = nn.Parameter(
                torch.Tensor(self.d_channel, 1, self.kernel_size),
                requires_grad=requires_grad,
            )
            self.h1 = nn.Parameter(
                torch.Tensor(self.d_channel, 1, self.kernel_size),
                requires_grad=requires_grad,
            )
            nn.init.xavier_uniform_(self.h0)
            nn.init.xavier_uniform_(self.h1)

            with torch.no_grad():
                self.h0.data = self.h0.data / torch.norm(
                    self.h0.data, dim=-1, keepdim=True
                )
                self.h1.data = self.h1.data / torch.norm(
                    self.h1.data, dim=-1, keepdim=True
                )

    def forward(self, x):
        if self.swt:
            coeffs = self.swt_decomposition(
                x, self.h0, self.h1, self.m, self.kernel_size
            )
        else:
            coeffs = self.swt_reconstruction(
                x, self.h0, self.h1, self.m, self.kernel_size
            )
        return coeffs

    def swt_decomposition(self, x, h0, h1, depth, kernel_size):
        approx_coeffs = x
        coeffs = []
        dilation = 1
        for _ in range(depth):
            padding = dilation * (kernel_size - 1)
            padding_r = (kernel_size * dilation) // 2
            pad = (padding - padding_r, padding_r)
            approx_coeffs_pad = F.pad(approx_coeffs, pad, "circular")
            detail_coeff = F.conv1d(
                approx_coeffs_pad, h1, dilation=dilation, groups=x.shape[1]
            )
            approx_coeffs = F.conv1d(
                approx_coeffs_pad, h0, dilation=dilation, groups=x.shape[1]
            )
            coeffs.append(detail_coeff)
            dilation *= 2
        coeffs.append(approx_coeffs)

        return torch.stack(list(reversed(coeffs)), -2)

    def swt_reconstruction(self, coeffs, g0, g1, m, kernel_size):
        dilation = 2 ** (m - 1)
        approx_coeff = coeffs[:, :, 0, :]
        detail_coeffs = coeffs[:, :, 1:, :]

        for i in range(m):
            detail_coeff = detail_coeffs[:, :, i, :]
            padding = dilation * (kernel_size - 1)
            padding_l = (dilation * kernel_size) // 2
            pad = (padding_l, padding - padding_l)
            approx_coeff_pad = F.pad(approx_coeff, pad, "circular")
            detail_coeff_pad = F.pad(detail_coeff, pad, "circular")

            y = F.conv1d(
                approx_coeff_pad, g0, groups=approx_coeff.shape[1], dilation=dilation
            ) + F.conv1d(
                detail_coeff_pad, g1, groups=detail_coeff.shape[1], dilation=dilation
            )
            approx_coeff = y / 2
            dilation //= 2

        return approx_coeff


class GeomAttention(nn.Module):
    def __init__(
        self,
        mask_flag=False,
        factor=5,
        scale=None,
        attention_dropout=0.1,
        output_attention=False,
        alpha=1.0,
    ):
        super(GeomAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

        self.alpha = alpha

    def forward(self, queries, keys, values, attn_mask=None):
        B, L, H, E = queries.shape
        _, S, _, _ = values.shape
        scale = self.scale or 1.0 / sqrt(E)

        dot_product = torch.einsum("blhe,bshe->bhls", queries, keys)

        queries_norm2 = torch.sum(queries**2, dim=-1)
        keys_norm2 = torch.sum(keys**2, dim=-1)
        queries_norm2 = queries_norm2.permute(0, 2, 1).unsqueeze(-1)  # (B, H, L, 1)
        keys_norm2 = keys_norm2.permute(0, 2, 1).unsqueeze(-2)  # (B, H, 1, S)
        wedge_norm2 = queries_norm2 * keys_norm2 - dot_product**2  # (B, H, L, S)
        wedge_norm2 = F.relu(wedge_norm2)
        wedge_norm = torch.sqrt(wedge_norm2 + 1e-8)

        scores = (1 - self.alpha) * dot_product + self.alpha * wedge_norm
        scores = scores * scale

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.tril(torch.ones(L, S)).to(scores.device)
            scores.masked_fill_(attn_mask.unsqueeze(1).unsqueeze(2) == 0, float("-inf"))

        A = self.dropout(torch.softmax(scores, dim=-1))

        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous()
        else:
            return (V.contiguous(), scores.abs().mean())


class GeomAttentionLayer(nn.Module):
    def __init__(
        self,
        attention: GeomAttention,
        d_model: int,
        requires_grad: bool = True,
        wv: str = "db2",
        m: int = 2,
        kernel_size: Tuple[int, None] = None,
        d_channel: Tuple[int, None] = None,
        geomattn_dropout: float = 0.5,
    ):
        super(GeomAttentionLayer, self).__init__()

        self.d_channel = d_channel
        self.inner_attention = attention

        self.swt = WaveletEmbedding(
            d_channel=self.d_channel,
            swt=True,
            requires_grad=requires_grad,
            wv=wv,
            m=m,
            kernel_size=kernel_size,
        )
        self.query_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
        )
        self.key_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
        )
        self.value_projection = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Dropout(geomattn_dropout)
        )
        self.out_projection = nn.Sequential(
            nn.Linear(d_model, d_model),
            WaveletEmbedding(
                d_channel=self.d_channel,
                swt=False,
                requires_grad=requires_grad,
                wv=wv,
                m=m,
                kernel_size=kernel_size,
            ),
        )

    def forward(self, queries, keys, values, attn_mask=None, tau=None, delta=None):
        queries = self.swt(queries)
        keys = self.swt(keys)
        values = self.swt(values)

        queries = self.query_projection(queries).permute(0, 3, 2, 1)
        keys = self.key_projection(keys).permute(0, 3, 2, 1)
        values = self.value_projection(values).permute(0, 3, 2, 1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
        )

        out = self.out_projection(out.permute(0, 3, 2, 1))

        return out, attn


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        return self.dropout(x)


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        n_channels: int = 7,
        d_model: int = 512,
        output_attention: bool = False,
        geomattn_dropout: float = 0.5,
        alpha: float = 1,
        kernel_size: int = 3,  # TODO
        embed: str = "timeF",
        freq: float = 1.0,
        dropout: float = 0.1,
        wv: str = "db1",
        factor: int = 1,
        requires_grad: bool = True,
        m: int = 3,
        d_ff: int = 32,
        activation: str = "gelu",
        e_layers: int = 1,
    ):
        super(Model, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.output_attention = output_attention
        self.geomattn_dropout = geomattn_dropout
        self.alpha = alpha
        self.kernel_size = kernel_size

        enc_embedding = DataEmbedding_inverted(
            seq_len,
            d_model,
            embed,
            freq,
            dropout,
        )
        self.enc_embedding = enc_embedding

        encoder = Encoder(
            [
                EncoderLayer(
                    GeomAttentionLayer(
                        GeomAttention(
                            False,
                            factor,
                            attention_dropout=dropout,
                            output_attention=output_attention,
                            alpha=self.alpha,
                        ),
                        d_model,
                        requires_grad=requires_grad,
                        wv=wv,
                        m=m,
                        d_channel=n_channels,
                        kernel_size=self.kernel_size,
                        geomattn_dropout=self.geomattn_dropout,
                    ),
                    d_model,
                    d_ff,
                    dropout=dropout,
                    activation=activation,
                )
                for _ in range(e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(d_model),
        )
        self.encoder = encoder

        projector = nn.Linear(d_model, self.pred_len, bias=True)
        self.projector = projector

    def forward(self, x_enc: Tensor, x_mark_enc: Tensor) -> Tuple[Tensor, Tensor]:
        _, _, N = x_enc.shape

        enc_embedding = self.enc_embedding
        encoder = self.encoder
        projector = self.projector
        # Linear Projection             B L N -> B L' (pseudo temporal tokens) N
        enc_out = enc_embedding(x_enc, x_mark_enc)

        # SimpleTM Layer                B L' N -> B L' N
        enc_out, attns = encoder(enc_out, attn_mask=None)

        # Output Projection             B L' N -> B H (Horizon) N
        dec_out = projector(enc_out).permute(0, 2, 1)[:, :, :N]

        return dec_out, attns


class SimpleTM(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        loss_fn: str = "MSE",
        learning_rate: float = 0.02,
        lradj: str = "TST",
        data: str = "custom",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss_fn)
        self.save_hyperparameters(ignore=["model", "criterion"])

    def model_specific_forward(self, look_back_window: Tensor) -> Tensor:
        # B, T, C = x.shape
        # time = torch.zeros((B, T, 5), dtype=float)
        preds, _ = self.model(
            look_back_window, None
        )  # None works for the time embedding (see DataEmbedding_inverted)
        return preds

    def _shared_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tuple[Tensor, Tensor]:
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        assert preds.shape == prediction_window.shape

        mae_criterion = torch.nn.L1Loss()
        mae_loss = mae_criterion(preds, prediction_window)
        loss = self.criterion(preds, prediction_window)
        return loss, mae_loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, _ = self._shared_step(look_back_window, prediction_window)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=True, on_epoch=True, logger=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def configure_optimizers(
        self,
    ) -> Union[
        torch.optim.Optimizer,
        Dict[str, Union[torch.optim.Adam, torch.optim.lr_scheduler.OneCycleLR]],
    ]:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        if self.hparams.lradj == "type1":
            steps_per_epoch = (
                self.trainer.estimated_stepping_batches // self.trainer.max_epochs
            )

            print(self.trainer.estimated_stepping_batches)
            print(self.trainer.max_epochs)

            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hparams.learning_rate,
                steps_per_epoch=steps_per_epoch,
                epochs=self.trainer.max_epochs,
                pct_start=0.2,
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "step",  # because OneCycleLR is stepped per batch
                "frequency": 1,
                "name": "OneCycleLR",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return optimizer
