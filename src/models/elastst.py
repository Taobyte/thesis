# Adapted from Microsoft's ProbTS project:
# https://github.com/microsoft/ProbTS
# Original license: MIT License
# Modifications made by Clemens Keusch, 14.04.2025

import re
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch import Tensor
from einops import rearrange, repeat
from typing import List, Union, Optional, Callable, Tuple

import lightning.pytorch as pl
import sys

from src.datasets.elastst.data_wrapper import ProbTSBatchData
from src.datasets.elastst.data_utils.data_scaler import Scaler, IdentityScaler


class Time_Encoder(nn.Module):
    def __init__(self, embed_time):
        super(Time_Encoder, self).__init__()
        self.periodic = nn.Linear(1, embed_time - 1)
        self.linear = nn.Linear(1, 1)

    def forward(self, tt):
        if tt.dim() == 3:  # [B,L,K]
            tt = rearrange(tt, "b l k -> b l k 1")
        else:  # [B,L]
            tt = rearrange(tt, "b l -> b l 1 1")

        out2 = torch.sin(self.periodic(tt))
        out1 = self.linear(tt)
        out = torch.cat([out1, out2], -1)  # [B,L,1,D]
        return out


def sin_cos_encoding(B, K, L, embed_dim):
    assert embed_dim % 2 == 0

    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    pos = [i for i in range(L)]
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)

    emb = repeat(emb, "l d -> b k l d", b=B, k=K)
    return torch.tensor(emb, dtype=torch.float64)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        seq_len: int,
        base: float = 10000.0,
        learnable=False,
        init="exp",
        min_period=0.01,
        max_period=1000,
    ):
        super(RotaryEmbedding, self).__init__()
        if init == "linear":
            theta = get_linear_period(min_period, max_period, dim)
        elif init == "uniform":
            theta = torch.ones([dim // 2])
            periods = torch.nn.init.uniform_(theta, a=min_period, b=max_period)
            theta = 2 * np.pi / periods
        elif init == "exp":
            theta = get_exp_period(min_period, max_period, dim)
        elif init == "rope":
            theta = 1.0 / (
                base ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)
            )
        else:
            print("invalid theta init")
            sys.exit(0)

        if learnable:
            self.freqs = nn.Parameter(theta)
        else:
            self.register_buffer("freqs", torch.tensor(theta))

        self.dim = dim
        self.seq_len = seq_len
        self.learnable = learnable

    def forward(self, xq: torch.Tensor, xk: torch.Tensor, xv: torch.Tensor):
        L = xq.shape[-2]
        t = torch.arange(L, device=xq.device)

        freqs = torch.outer(t, self.freqs).float()  # m * \theta
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
        xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
        xv_ = xv.float().reshape(*xv.shape[:-1], -1, 2)

        xq_ = torch.view_as_complex(xq_).to(xq.device)
        xk_ = torch.view_as_complex(xk_).to(xq.device)
        xv_ = torch.view_as_complex(xv_).to(xq.device)

        # rotate and then map to real number field
        xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2).to(xq.device)
        xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2).to(xq.device)
        xv_out = torch.view_as_real(xv_ * freqs_cis).flatten(2).to(xq.device)
        return xq_out.type_as(xq), xk_out.type_as(xk), xv_out.type_as(xv)


def get_linear_period(min_period, max_period, dim):
    i = torch.arange(0, dim, 2)[: (dim // 2)]

    periods = min_period + ((max_period - min_period) / dim) * i
    theta = 2 * np.pi / periods
    return theta


def get_exp_period(min_period, max_period, dim):
    i = torch.arange(0, dim, 2)[: (dim // 2)]
    max_theta = 2 * np.pi / min_period
    min_theta = 2 * np.pi / max_period
    alpha = np.log(max_theta / min_theta) * (1 / (dim - 2))
    thetas = max_theta * np.exp(-alpha * i)
    return thetas


# generate rotation matrix
def precompute_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # rotate \theta_i
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # generate token indexes t = [0, 1,..., seq_len-1]
    t = torch.arange(seq_len, device=freqs.device)
    # freqs.shape = [seq_len, dim // 2]
    freqs = torch.outer(t, freqs).float()  # m * \theta

    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    xv: torch.Tensor,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 2)
    xv_ = xv.float().reshape(*xv.shape[:-1], -1, 2)

    freqs_cis = freqs_cis.to(xq.device)

    xq_ = torch.view_as_complex(xq_).to(xq.device)
    xk_ = torch.view_as_complex(xk_).to(xq.device)
    xv_ = torch.view_as_complex(xv_).to(xq.device)

    # rotate and then map to real number field
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2).to(xq.device)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2).to(xq.device)
    xv_out = torch.view_as_real(xv_ * freqs_cis).flatten(2).to(xq.device)
    return xq_out.type_as(xq), xk_out.type_as(xk), xv_out.type_as(xv)


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.2):
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.bmm(q / self.temperature, k.transpose(-2, -1))

        if mask is not None and mask.dim() == 5:
            mask = mask.transpose(2, 4)

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.bmm(attn, v)

        return output, attn


class ScaledDotProductAttention_bias(nn.Module):
    def __init__(
        self,
        d_model,
        n_head,
        d_k,
        d_v,
        temperature,
        attn_dropout=0.2,
        rotate=False,
        max_seq_len=100,
        theta=10000,
        addv=False,
        learnable_theta=False,
        bin_att=False,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super().__init__()

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)
        self.n_head = n_head
        self.bin_att = bin_att
        self.rotate = rotate
        self.addv = addv
        self.trope = RotaryEmbedding(
            d_v,
            max_seq_len,
            base=theta,
            learnable=learnable_theta,
            init=rope_theta_init,
            min_period=min_period,
            max_period=max_period,
        )

        if self.bin_att:
            self.alpha = nn.Parameter(torch.zeros([1, 1, n_head, 1, 1]))
            self.beta = nn.Parameter(torch.zeros([1, 1, n_head, 1, 1]))

    def forward(self, q, k, v, mask):
        # input: [B,K,H,LQ,LK] for temporal, [B,L,H,Kq,Kk] for category

        # [B,K,L,H,D]
        q = rearrange(self.w_qs(q), "b k l (n d) -> b k n l d", n=self.n_head)
        k = rearrange(self.w_ks(k), "b k l (n d) -> b k n d l", n=self.n_head)
        v = rearrange(self.w_vs(v), "b k l (n d) -> b k n l d", n=self.n_head)

        B, K, N, L, D = q.shape
        if self.rotate:
            xq = rearrange(q, "b k n l d -> (b k n) l d")
            xk = rearrange(k, "b k n d l -> (b k n) l d")
            xv = rearrange(v, "b k n l d -> (b k n) l d")

            xq, xk, xv = self.trope(xq, xk, xv)

            attn = torch.matmul(xq, xk.transpose(1, 2)) / self.temperature
            attn = rearrange(attn, "(b k n) l t -> b k n l t", b=B, k=K, n=N)
            if self.addv:
                v = rearrange(xv, "(b k n) l d -> b k n l d", b=B, k=K, n=N)
        else:
            attn = torch.matmul(q, k) / self.temperature

        if self.bin_att:
            self_mask = torch.eye(L).to(mask.device)
            self_mask = repeat(self_mask, "l t -> b k n l t", b=B, k=K, n=N)

            attn = attn + self_mask * self.alpha + (1 - self_mask) * self.beta

        if mask is not None:
            if attn.dim() > mask.dim():
                mask = mask.unsqueeze(2).expand(attn.shape)
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))

        v = torch.matmul(attn, v)

        v = rearrange(v, "b k n l d -> b k l (n d)")

        # sys.exit(0)
        return v, attn


class Attention(nn.Module):
    def __init__(self, hin_d, d_model):
        super().__init__()

        self.linear = nn.Linear(d_model, hin_d)
        self.W = nn.Linear(hin_d, 1, bias=False)

    def forward(self, x, mask=None, mask_value=-1e30):
        # [B,K,L,D]

        # map directly
        attn = self.W(torch.tanh(self.linear(x)))  # [B,K,L,1]

        if mask is not None:
            attn = mask * attn + (1 - mask) * mask_value

        attn = F.softmax(attn, dim=-2)

        x = torch.matmul(x.transpose(-1, -2), attn).squeeze(-1)  # [B,K,D,1]

        return x, attn


class MultiHeadAttention_tem_bias(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        n_head,
        d_model,
        d_k,
        d_v,
        dropout=0.1,
        rotate=False,
        max_seq_len=100,
        theta=10000,
        addv=False,
        learnable_theta=False,
        bin_att=False,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)

        self.attention = ScaledDotProductAttention_bias(
            d_model,
            n_head,
            d_k,
            d_v,
            temperature=d_k**0.5,
            attn_dropout=dropout,
            rotate=rotate,
            max_seq_len=max_seq_len,
            theta=theta,
            addv=addv,
            learnable_theta=learnable_theta,
            bin_att=bin_att,
            rope_theta_init=rope_theta_init,
            min_period=min_period,
            max_period=max_period,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # event_matrix [B,L,K]

        # [B,K,H,Lq,Lk]
        output, attn = self.attention(q, k, v, mask=mask)  # [B,K,H,L,D]

        output = self.dropout(self.fc(output))

        return output, attn


class MultiHeadAttention_type_bias(nn.Module):
    """Multi-Head Attention module"""

    def __init__(
        self,
        n_head,
        d_model,
        d_k,
        d_v,
        dropout=0.1,
        rotate=False,
        max_seq_len=1024,
        bin_att=False,
    ):
        super().__init__()
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.fc = nn.Linear(d_v * n_head, d_model)
        self.attention = ScaledDotProductAttention_bias(
            d_model,
            n_head,
            d_k,
            d_v,
            temperature=d_k**0.5,
            attn_dropout=dropout,
            rotate=False,
            max_seq_len=max_seq_len,
            bin_att=bin_att,
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # [B,L,K,D]
        output, attn = self.attention(q, k, v, mask=mask)

        output = self.dropout(self.fc(output))

        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)

        return x


PAD = 0


def get_attn_key_pad_mask_K(
    past_value_indicator, observed_indicator, transpose=False, structured_mask=False
):
    """For masking out the padding part of key sequence.
    input: mask: transpose=False: [b k l]
    """

    if structured_mask:
        mask = past_value_indicator
    else:
        mask = observed_indicator

    if transpose:
        mask = rearrange(mask, "b l k -> b k l")
    padding_mask = repeat(mask, "b k l1 -> b k l2 l1", l2=mask.shape[-1]).eq(PAD)

    return padding_mask


class EncoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self,
        d_model,
        d_inner,
        n_head,
        d_k,
        d_v,
        dropout=0.1,
        tem_att=True,
        type_att=False,
        structured_mask=True,
        rotate=False,
        max_seq_len=100,
        theta=10000,
        addv=False,
        learnable_theta=False,
        bin_att=False,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super(EncoderLayer, self).__init__()

        self.structured_mask = structured_mask
        self.tem_att = tem_att
        self.type_att = type_att

        if tem_att:
            self.slf_tem_attn = MultiHeadAttention_tem_bias(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                rotate=rotate,
                max_seq_len=max_seq_len,
                theta=theta,
                addv=addv,
                learnable_theta=learnable_theta,
                bin_att=bin_att,
                rope_theta_init=rope_theta_init,
                min_period=min_period,
                max_period=max_period,
            )

        if type_att:
            self.slf_type_attn = MultiHeadAttention_type_bias(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                rotate=False,
                max_seq_len=max_seq_len,
                bin_att=bin_att,
            )

        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, input, past_value_indicator=None, observed_indicator=None):
        # time attention
        # [B, K, L, D]
        if self.tem_att:
            tem_mask = get_attn_key_pad_mask_K(
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
                transpose=False,
                structured_mask=self.structured_mask,
            )
            tem_output = self.layer_norm(input)

            tem_output, enc_tem_attn = self.slf_tem_attn(
                tem_output, tem_output, tem_output, mask=tem_mask
            )

            tem_output = tem_output + input
        else:
            tem_output = input

        tem_output = rearrange(tem_output, "b k l d -> b l k d")

        # type attention
        # [B, L, K, D]
        if self.type_att:
            type_mask = get_attn_key_pad_mask_K(
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
                transpose=True,
                structured_mask=self.structured_mask,
            )

            type_output = self.layer_norm(tem_output)

            type_output, enc_type_attn = self.slf_type_attn(
                type_output, type_output, type_output, mask=type_mask
            )

            enc_output = type_output + tem_output
        else:
            enc_output = tem_output

        # FFNN
        output = self.layer_norm(enc_output)

        output = self.pos_ffn(output)

        output = output + enc_output

        output = rearrange(output, "b l k d -> b k l d")

        # optional
        output = self.layer_norm(output)

        return output  # , enc_tem_attn, enc_type_attn


def convert_to_list(s):
    """
    Convert prediction length strings into list
    e.g., '96-192-336-720' will be convert into [96,192,336,720]
    Input: str, list, int
    Returns: list
    """
    if type(s).__name__ == "int":
        return [s]
    elif type(s).__name__ == "list":
        return s
    elif type(s).__name__ == "str":
        elements = re.split(r"\D+", s)
        return list(map(int, elements))
    else:
        return None


def weighted_average(
    x: torch.Tensor,
    weights: Optional[torch.Tensor] = None,
    dim: int = None,
    reduce: str = "mean",
):
    """
    Computes the weighted average of a given tensor across a given dim, masking
    values associated with weight zero,
    meaning instead of `nan * 0 = nan` you will get `0 * 0 = 0`.

    Args:
        x: Input tensor, of which the average must be computed.
        weights: Weights tensor, of the same shape as `x`.
        dim: The dim along which to average `x`

    Returns:
        Tensor: The tensor with values averaged along the specified `dim`.
    """
    if weights is not None:
        weighted_tensor = torch.where(weights != 0, x * weights, torch.zeros_like(x))
        if reduce != "mean":
            return weighted_tensor
        sum_weights = torch.clamp(
            weights.sum(dim=dim) if dim else weights.sum(), min=1.0
        )
        return (
            weighted_tensor.sum(dim=dim) if dim else weighted_tensor.sum()
        ) / sum_weights
    else:
        return x.mean(dim=dim) if dim else x


class TemporalScaler(Scaler):
    def __init__(self, minimum_scale: float = 1e-10, time_first: bool = True):
        """
        The ``TemporalScaler`` computes a per-item scale according to the average
        absolute value over time of each item. The average is computed only among
        the observed values in the data tensor, as indicated by the second
        argument. Items with no observed data are assigned a scale based on the
        global average.

        Args:
            minimum_scale: default scale that is used if the time series has only zeros.
            time_first: if True, the input tensor has shape (N, T, C), otherwise (N, C, T).
        """
        super().__init__()
        self.scale = None
        self.minimum_scale = torch.tensor(minimum_scale)
        self.time_first = time_first

    def fit(self, data: torch.Tensor, observed_indicator: torch.Tensor = None):
        """
        Fit the scaler to the data.

        Args:
            data: tensor of shape (N, T, C) if ``time_first == True`` or (N, C, T)
                if ``time_first == False`` containing the data to be scaled

            observed_indicator: observed_indicator: binary tensor with the same shape as
                ``data``, that has 1 in correspondence of observed data points,
                and 0 in correspondence of missing data points.

        Note:
            Tensor containing the scale, of shape (N, 1, C) or (N, C, 1).
        """
        if self.time_first:
            dim = -2
        else:
            dim = -1

        if observed_indicator is None:
            observed_indicator = torch.ones_like(data)

        # These will have shape (N, C)
        num_observed = observed_indicator.sum(dim=dim)
        sum_observed = (data.abs() * observed_indicator).sum(dim=dim)

        # First compute a global scale per-dimension
        total_observed = num_observed.sum(dim=0)
        denominator = torch.max(total_observed, torch.ones_like(total_observed))
        default_scale = sum_observed.sum(dim=0) / denominator

        # Then compute a per-item, per-dimension scale
        denominator = torch.max(num_observed, torch.ones_like(num_observed))
        scale = sum_observed / denominator

        # Use per-batch scale when no element is observed
        # or when the sequence contains only zeros
        scale = torch.where(
            sum_observed > torch.zeros_like(sum_observed),
            scale,
            default_scale * torch.ones_like(num_observed),
        )

        self.scale = torch.max(scale, self.minimum_scale).unsqueeze(dim=dim).detach()

    def transform(self, data):
        return data / self.scale.to(data.device)

    def fit_transform(self, data, observed_indicator=None):
        self.fit(data, observed_indicator)
        return self.transform(data)

    def inverse_transform(self, data):
        return data * self.scale.to(data.device)


class InstanceNorm(nn.Module):
    def __init__(self, eps=1e-5):
        """
        :param eps: a value added for numerical stability
        """
        super(InstanceNorm, self).__init__()
        self.eps = eps

    def forward(self, x, mode: str):
        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        return x

    def _denormalize(self, x):
        x = x * self.stdev
        x = x + self.mean
        return x


# Cell
class ElasTST_backbone(nn.Module):
    def __init__(
        self,
        l_patch_size: list,
        stride: int = None,
        k_patch_size: int = 1,
        in_channels: int = 1,
        n_layers: int = 0,
        t_layers: int = 1,
        v_layers: int = 1,
        hidden_size: int = 256,
        n_heads: int = 16,
        d_k: Optional[int] = None,
        d_v: Optional[int] = None,
        d_inner: int = 256,
        dropout: float = 0.0,
        rotate: bool = False,
        max_seq_len=1000,
        theta=10000,
        learnable_theta=False,
        addv: bool = False,
        bin_att: bool = False,
        abs_tem_emb: bool = False,
        learn_tem_emb: bool = False,
        structured_mask: bool = True,
        rope_theta_init: str = "exp",
        min_period: float = 1,
        max_period: float = 1000,
        patch_share_backbone: bool = True,
    ):
        super().__init__()

        if rotate:
            print(
                f"Using Rotary Embedding... [theta init]: {rope_theta_init}, [period range]: [{min_period},{max_period}], [learnable]: {learnable_theta}"
            )
        print(
            "[Binary Att.]: ",
            bin_att,
            " [Learned time emb]: ",
            learn_tem_emb,
            " [Abs time emb]: ",
            abs_tem_emb,
        )
        print("[Multi Patch Share Backbone]: ", patch_share_backbone)
        print("[Structured Mask]: ", not structured_mask)
        # Patching
        self.l_patch_size = l_patch_size
        self.k_patch_size = k_patch_size
        self.in_channels = in_channels
        self.out_channels = in_channels
        self.patch_share_backbone = patch_share_backbone
        self.abs_tem_emb = abs_tem_emb

        self.hidden_size = hidden_size
        if stride is not None:
            self.stride = stride
        else:
            self.stride = self.l_patch_size

        x_embedder = []
        final_layer = []
        backbone = []
        for p in self.l_patch_size:
            print(f"=== Patch {p} Branch ===")
            x_embedder.append(
                TimePatchEmbed(
                    p,
                    self.k_patch_size,
                    self.in_channels,
                    self.hidden_size,
                    bias=True,
                    stride=p,
                )
            )
            final_layer.append(
                MLP_FinalLayer(
                    self.hidden_size, p, self.k_patch_size, self.out_channels
                )
            )

            if not patch_share_backbone:
                backbone.append(
                    DoublyAtt(
                        d_model=self.hidden_size,
                        n_layers=n_layers,
                        t_layers=t_layers,
                        v_layers=v_layers,
                        d_inner=d_inner,
                        n_heads=n_heads,
                        d_k=d_k,
                        d_v=d_v,
                        dropout=dropout,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        bin_att=bin_att,
                        learnable_theta=learnable_theta,
                        structured_mask=structured_mask,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )

        self.x_embedder = nn.ModuleList(x_embedder)
        self.final_layer = nn.ModuleList(final_layer)

        if not patch_share_backbone:
            self.backbone = nn.ModuleList(backbone)
        else:
            self.backbone = DoublyAtt(
                d_model=self.hidden_size,
                n_layers=n_layers,
                t_layers=t_layers,
                v_layers=v_layers,
                d_inner=d_inner,
                n_heads=n_heads,
                d_k=d_k,
                d_v=d_v,
                dropout=dropout,
                rotate=rotate,
                max_seq_len=max_seq_len,
                theta=theta,
                addv=addv,
                bin_att=bin_att,
                learnable_theta=learnable_theta,
                structured_mask=structured_mask,
                rope_theta_init=rope_theta_init,
                min_period=min_period,
                max_period=max_period,
            )

        self.learn_tem_emb = learn_tem_emb
        if self.learn_tem_emb:
            self.learn_time_embedding = Time_Encoder(self.hidden_size)

    def get_patch_num(self, dim_size, len_size, l_patch_size):
        num_k_patches = int((dim_size - self.k_patch_size) / self.k_patch_size + 1)
        num_l_patches = int((len_size - l_patch_size) / l_patch_size + 1)
        return num_k_patches, num_l_patches

    def forward(
        self,
        past_target,
        future_placeholder,
        past_observed_values,
        future_observed_values,
        dataset_name=None,
    ):  # z: [bs x nvars x seq_len]
        pred_shape = future_placeholder.shape
        future_observed_indicator = torch.zeros(future_observed_values.shape).to(
            future_observed_values.device
        )

        x = torch.cat((past_target, future_placeholder), dim=1)  # B L+T K

        past_value_indicator = torch.cat(
            (past_observed_values, future_observed_indicator), dim=1
        )  # B L+T K
        observed_value_indicator = torch.cat(
            (past_observed_values, future_observed_values), dim=1
        )  # B L+T K

        pred_list = []

        for idx in range(len(self.l_patch_size)):
            x_p = x.clone()

            num_k_patches, num_l_patches = self.get_patch_num(
                x_p.shape[-1], x_p.shape[-2], self.l_patch_size[idx]
            )

            # do patching
            x_p, past_value_indicator_p, observed_value_indicator_p = self.x_embedder[
                idx
            ](x_p, past_value_indicator, observed_value_indicator)  # b k l d

            if self.learn_tem_emb:
                grid_len = np.arange(num_l_patches, dtype=np.float32)
                grid_len = (
                    torch.tensor(grid_len, requires_grad=False)
                    .float()
                    .unsqueeze(0)
                    .to(x.device)
                )
                pos_embed = repeat(grid_len, "1 l -> b l", b=pred_shape[0])
                pos_embed = self.learn_time_embedding(pos_embed)  # b l 1 d
                pos_embed = rearrange(pos_embed, "b l 1 d -> b 1 l d")
                x_p = x_p + pos_embed

            # use a absolute position embedding
            if self.abs_tem_emb:
                B, K, L, embed_dim = x_p.shape
                pos_embed = sin_cos_encoding(B, K, L, embed_dim).float()  # b k l d
                x_p = x_p + pos_embed.to(x_p.device)

            # model
            if self.patch_share_backbone:
                x_p = self.backbone(
                    x_p, past_value_indicator_p, observed_value_indicator_p
                )  # b k l d
            else:
                x_p = self.backbone[idx](
                    x_p, past_value_indicator_p, observed_value_indicator_p
                )  # b k l d

            x_p = self.final_layer[idx](x_p)  # b k l p

            x_p = rearrange(x_p, "b k t p -> b (t p) k")

            x_p = x_p[:, -pred_shape[1] :, :]

            pred_list.append(x_p.unsqueeze(-1))

        pred_list = torch.cat(pred_list, dim=-1)
        multi_patch_mean_res = torch.mean(pred_list, dim=-1)

        return multi_patch_mean_res, pred_list


class DoublyAtt(nn.Module):
    def __init__(
        self,
        d_model,
        n_layers,
        d_inner,
        n_heads,
        d_k,
        d_v,
        dropout,
        rotate=False,
        max_seq_len=1024,
        theta=10000,
        t_layers=2,
        v_layers=1,
        bin_att=False,
        addv=False,
        learnable_theta=False,
        structured_mask=True,
        rope_theta_init="exp",
        min_period=0.1,
        max_period=10,
    ):
        super().__init__()
        # assert n_layers <= (t_layers + v_layers) <= 2*n_layers , "Sum of t_layers and n_layers must be between 1 and 2"

        # Configuration based on temporal and variate ratios
        self.layer_stack = nn.ModuleList()
        num_t = t_layers
        num_v = v_layers
        num_both = min(t_layers, v_layers)

        num_t = num_t - num_both
        num_v = num_v - num_both

        t_count = 0
        v_count = 0
        for _ in range(num_t + num_v):
            if t_count < num_t:
                self.layer_stack.append(
                    EncoderLayer(
                        d_model,
                        d_inner,
                        n_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        tem_att=True,
                        type_att=False,
                        structured_mask=structured_mask,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        learnable_theta=learnable_theta,
                        bin_att=bin_att,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )
                t_count = t_count + 1
                print(f"[Encoder Layer {t_count + v_count}] Use tem att")
            if v_count < num_v:
                self.layer_stack.append(
                    EncoderLayer(
                        d_model,
                        d_inner,
                        n_heads,
                        d_k,
                        d_v,
                        dropout=dropout,
                        tem_att=False,
                        type_att=True,
                        structured_mask=structured_mask,
                        rotate=rotate,
                        max_seq_len=max_seq_len,
                        theta=theta,
                        addv=addv,
                        learnable_theta=learnable_theta,
                        bin_att=bin_att,
                        rope_theta_init=rope_theta_init,
                        min_period=min_period,
                        max_period=max_period,
                    )
                )
                v_count = v_count + 1
                print(f"[Encoder Layer {t_count + v_count}] Use var att")

        for idx in range(num_both):
            self.layer_stack.append(
                EncoderLayer(
                    d_model,
                    d_inner,
                    n_heads,
                    d_k,
                    d_v,
                    dropout=dropout,
                    tem_att=True,
                    type_att=True,
                    structured_mask=structured_mask,
                    rotate=rotate,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    addv=addv,
                    learnable_theta=learnable_theta,
                    bin_att=bin_att,
                    rope_theta_init=rope_theta_init,
                    min_period=min_period,
                    max_period=max_period,
                )
            )

            print(f"[Encoder Layer {idx + t_count + v_count}] Use tem and var att")

    def forward(self, x, past_value_indicator, observed_indicator) -> Tensor:
        for enc_layer in self.layer_stack:
            x = enc_layer(
                x,
                past_value_indicator=past_value_indicator,
                observed_indicator=observed_indicator,
            )

        return x


class MLP_FinalLayer(nn.Module):
    """
    The final layer of DiT.
    """

    def __init__(self, hidden_size, l_patch_size, k_patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(
            hidden_size, l_patch_size * k_patch_size * out_channels, bias=True
        )

    def forward(self, x):
        x = self.norm_final(x)
        x = self.linear(x)
        return x


class TimePatchEmbed(nn.Module):
    """Time Patch Embedding"""

    def __init__(
        self,
        l_patch_size: int = 16,
        k_patch_size=1,
        in_chans: int = 1,
        embed_dim: int = 768,
        norm_layer: Optional[Callable] = None,
        flatten: bool = False,
        bias: bool = True,
        # padding_patch = None,
        stride=None,
        # strict_img_size: bool = True,
    ):
        super().__init__()
        self.l_patch_size = l_patch_size
        self.k_patch_size = k_patch_size
        if stride is None:
            stride = l_patch_size

        self.flatten = flatten

        padding = 0
        kernel_size = (l_patch_size, k_patch_size)
        stride_size = (stride, k_patch_size)

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=kernel_size,
            stride=stride_size,
            bias=bias,
            padding=padding,
        )
        self.mask_proj = nn.Conv2d(
            1,
            1,
            kernel_size=kernel_size,
            stride=stride_size,
            bias=False,
            padding=padding,
        )

        self.mask_proj.weight.data.fill_(1.0)

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x, future_mask, obv_mask):
        """
        future_mask: only past values are set to 1
        obv_mask: past values and values to be predicted are set to 1
        """

        # B, C, K, L = x.shape
        if len(x.shape) == 3:
            x = rearrange(x, "b l k -> b 1 l k")

        future_mask = rearrange(future_mask, "b l k -> b 1 l k")
        obv_mask = rearrange(obv_mask, "b l k -> b 1 l k")

        x = self.proj(x)  # B C L K -> B C L' K

        with torch.no_grad():
            future_mask = self.mask_proj(future_mask)
            obv_mask = self.mask_proj(obv_mask)

        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
            future_mask = future_mask.flatten(2).transpose(1, 2)  # NCHW -> NLC
            obv_mask = obv_mask.flatten(2).transpose(1, 2)  # NCHW -> NLC

        x = self.norm(x)

        x = rearrange(x, "b d l k -> b k l d")
        future_mask = rearrange(future_mask, "b 1 l k -> b k l")
        obv_mask = rearrange(obv_mask, "b 1 l k -> b k l")
        return x, future_mask, obv_mask


class Forecaster(nn.Module):
    def __init__(
        self,
        target_dim: int,
        context_length: Union[list, int],
        prediction_length: Union[list, int],
        freq: str,
        use_lags: bool = False,
        use_feat_idx_emb: bool = False,
        use_time_feat: bool = False,
        feat_idx_emb_dim: int = 1,
        time_feat_dim: int = 1,
        use_scaling: bool = False,
        autoregressive: bool = False,
        no_training: bool = False,
        dataset: str = None,
        lags_list: List[int] = [64],  # TODO CHANGE THIS TO DYNAMICALLY ADD LAGS
        **kwargs,
    ):
        super().__init__()

        self.context_length = context_length
        self.prediction_length = prediction_length

        if isinstance(self.context_length, list):
            self.max_context_length = max(self.context_length)
        else:
            self.max_context_length = self.context_length

        if isinstance(self.prediction_length, list):
            self.max_prediction_length = max(self.prediction_length)
        else:
            self.max_prediction_length = self.prediction_length

        self.target_dim = target_dim
        self.freq = freq
        self.use_lags = use_lags
        self.use_feat_idx_emb = use_feat_idx_emb
        self.use_time_feat = use_time_feat
        self.feat_idx_emb_dim = feat_idx_emb_dim
        self.time_feat_dim = time_feat_dim
        self.autoregressive = autoregressive
        self.no_training = no_training
        self.use_scaling = use_scaling
        self.dataset = dataset
        # Lag parameters
        self.lags_list = lags_list
        if self.use_scaling:
            self.scaler = TemporalScaler()
        else:
            self.scaler = None

        self.lags_dim = len(self.lags_list) * target_dim
        self.feat_idx_emb = nn.Embedding(
            num_embeddings=self.target_dim, embedding_dim=self.feat_idx_emb_dim
        )
        self.input_size = self.get_input_size()

    @property
    def name(self):
        return self.__class__.__name__

    def get_input_size(self):
        input_size = self.target_dim if not self.use_lags else self.lags_dim
        if self.use_feat_idx_emb:
            input_size += self.use_feat_idx_emb * self.target_dim
        if self.use_time_feat:
            input_size += self.time_feat_dim
        return input_size

    def get_lags(self, sequence, lags_list, lags_length=1):
        """
        Get several lags from the sequence of shape (B, L, C) to (B, L', C*N),
        where L' = lag_length and N = len(lag_list).
        """
        assert max(lags_list) + lags_length <= sequence.shape[1]

        lagged_values = []
        for lag_index in lags_list:
            begin_index = -lag_index - lags_length
            end_index = -lag_index if lag_index > 0 else None
            lagged_value = sequence[:, begin_index:end_index, ...]
            if self.use_scaling:
                lagged_value = lagged_value / self.scaler.scale
            lagged_values.append(lagged_value)
        return torch.cat(lagged_values, dim=-1)

    def get_input_sequence(self, past_target_cdf, future_target_cdf, mode):
        if mode == "all":
            sequence = torch.cat((past_target_cdf, future_target_cdf), dim=1)
            seq_length = self.max_context_length + self.max_prediction_length
        elif mode == "encode":
            sequence = past_target_cdf
            seq_length = self.max_context_length
        elif mode == "decode":
            sequence = past_target_cdf
            seq_length = 1
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

        if self.use_lags:
            input_seq = self.get_lags(sequence, self.lags_list, seq_length)
        else:
            input_seq = sequence[:, -seq_length:, ...]
            if self.use_scaling:
                input_seq = input_seq / self.scaler.scale
        return input_seq

    def get_input_feat_idx_emb(self, target_dimension_indicator, input_length):
        input_feat_idx_emb = self.feat_idx_emb(target_dimension_indicator)  # [B K D]

        input_feat_idx_emb = (
            input_feat_idx_emb.unsqueeze(1)
            .expand(-1, input_length, -1, -1)
            .reshape(-1, input_length, self.target_dim * self.feat_idx_emb_dim)
        )
        return input_feat_idx_emb  # [B L K*D]

    def get_input_time_feat(self, past_time_feat, future_time_feat, mode):
        if mode == "all":
            time_feat = torch.cat(
                (past_time_feat[:, -self.max_context_length :, ...], future_time_feat),
                dim=1,
            )
        elif mode == "encode":
            time_feat = past_time_feat[:, -self.max_context_length :, ...]
        elif mode == "decode":
            time_feat = future_time_feat
        return time_feat

    def get_inputs(self, batch_data, mode):
        inputs_list = []

        input_seq = self.get_input_sequence(
            batch_data.past_target_cdf, batch_data.future_target_cdf, mode=mode
        )
        input_length = input_seq.shape[1]  # [B L n_lags*K]
        inputs_list.append(input_seq)

        if self.use_feat_idx_emb:
            input_feat_idx_emb = self.get_input_feat_idx_emb(
                batch_data.target_dimension_indicator, input_length
            )  # [B L K*D]
            inputs_list.append(input_feat_idx_emb)

        if self.use_time_feat:
            input_time_feat = self.get_input_time_feat(
                batch_data.past_time_feat, batch_data.future_time_feat, mode=mode
            )  # [B L Dt]
            inputs_list.append(input_time_feat)
        return torch.cat(inputs_list, dim=-1).to(dtype=torch.float32)

    def get_scale(self, batch_data):
        self.scaler.fit(
            batch_data.past_target_cdf[:, -self.max_context_length :, ...],
            batch_data.past_observed_values[:, -self.max_context_length :, ...],
        )

    def get_weighted_loss(self, batch_data, loss):
        observed_values = batch_data.future_observed_values
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1)
        return loss

    def loss(self, batch_data):
        raise NotImplementedError

    def forecast(self, batch_data=None, num_samples=None):
        raise NotImplementedError


class Model(Forecaster):
    def __init__(
        self,
        l_patch_size: Union[str, int, list] = "8_16_32",
        k_patch_size: int = 1,
        stride: int = None,
        rotate: bool = True,
        addv: bool = False,
        bin_att: bool = False,
        rope_theta_init: str = "exp",
        min_period: float = 1,
        max_period: float = 1000,
        learn_tem_emb: bool = False,
        learnable_rope: bool = True,
        abs_tem_emb: bool = False,
        structured_mask: bool = True,
        max_seq_len: int = 1024,
        theta_base: float = 10000,
        t_layers: int = 1,
        v_layers: int = 0,
        patch_share_backbone: bool = True,
        n_heads: int = 16,
        d_k: int = 8,
        d_v: int = 8,
        d_inner: int = 256,
        dropout: float = 0.0,
        in_channels: int = 1,
        f_hidden_size: int = 40,
        use_norm: bool = True,
        **kwargs,
    ):
        """
        ElasTST model.

        Parameters
        ----------
        l_patch_size : Union[str, int, list]
            Patch sizes configuration.
        k_patch_size : int
            Patch size for variables.
        stride : int
            Stride for patch splitting. If None, uses patch size as default.
        rotate : bool
            Apply rotational positional embeddings.
        addv : bool
            Whether to add RoPE information to value in attention. If False, only rotate the key and query embeddings.
        bin_att : bool
            Use binary attention biases to encode variate indices (any-variate attention).
        rope_theta_init : str
            Initialization for TRoPE, default is 'exp', as used in the paper. Options: ['exp', 'linear', 'uniform', 'rope'].
        min_period : float
            Minimum initialized period coefficient for rotary embeddings.
        max_period : float
            Maximum initialized period coefficient for rotary embeddings.
        learn_tem_emb : bool
            Whether to use learnable temporal embeddings.
        learnable_rope : bool
            Make period coefficient in TRoPE learnable.
        abs_tem_emb : bool
            Use absolute temporal embeddings if True.
        structured_mask : bool
            Apply structured mask or not.
        max_seq_len : int
            Maximum sequence length for the input time series.
        theta_base : int
            Base frequency of vanilla RoPE.
        t_layers : int
            Number of temporal attention layers.
        v_layers : int
            Number of variable attention layers.
        patch_share_backbone : bool
            Share Transformer backbone across patches.
        n_heads : int
            Number of attention heads in the multi-head attention mechanism.
        d_k : int
            Dimensionality of key embeddings in attention.
        d_v : int
            Dimensionality of value embeddings in attention.
        d_inner : int
            Size of inner layers in the feed-forward network.
        dropout : float
            Dropout rate for regularization during training.
        in_channels : int
            Number of input channels in the time series data. We only consider univariable.
        f_hidden_size : int
            Hidden size for the feed-forward layers.
        use_norm : bool
            Whether to apply instance normalization.
        **kwargs : dict
            Additional keyword arguments for extended functionality.
        """

        super().__init__(**kwargs)

        self.l_patch_size = convert_to_list(l_patch_size)
        self.use_norm = use_norm
        # Model
        self.model = ElasTST_backbone(
            l_patch_size=self.l_patch_size,
            stride=stride,
            k_patch_size=k_patch_size,
            in_channels=in_channels,
            t_layers=t_layers,
            v_layers=v_layers,
            hidden_size=f_hidden_size,
            d_inner=d_inner,
            n_heads=n_heads,
            d_k=d_k,
            d_v=d_v,
            dropout=dropout,
            rotate=rotate,
            max_seq_len=max_seq_len,
            theta=theta_base,
            addv=addv,
            bin_att=bin_att,
            learn_tem_emb=learn_tem_emb,
            abs_tem_emb=abs_tem_emb,
            learnable_theta=learnable_rope,
            structured_mask=structured_mask,
            rope_theta_init=rope_theta_init,
            min_period=min_period,
            max_period=max_period,
            patch_share_backbone=patch_share_backbone,
        )

        self.loss_fn = nn.MSELoss(reduction="none")
        self.instance_norm = InstanceNorm()

    def forward(self, batch_data, pred_len, dataset_name=None):
        new_pred_len = pred_len
        for p in self.l_patch_size:
            new_pred_len = self.check_divisibility(new_pred_len, p)

        look_back_window, prediction_window = batch_data
        past_target = look_back_window
        B, _, K = look_back_window.shape
        # B, _, K = batch_data.past_target_cdf.shape
        # past_target = batch_data.past_target_cdf
        # past_observed_values = batch_data.past_observed_values
        past_observed_values = torch.ones_like(look_back_window)

        if self.use_norm:
            past_target = self.instance_norm(past_target, "norm")

        # future_observed_values is the mask indicate whether there is a value in a position
        future_observed_values = torch.zeros([B, new_pred_len, K]).to(
            prediction_window.device
        )

        # pred_len = batch_data.future_observed_values.shape[1]
        # future_observed_values[:, :new_pred_len] = batch_data.future_observed_values

        # target placeholder
        future_placeholder = torch.zeros([B, new_pred_len, K]).to(
            prediction_window.device
        )

        x, pred_list = self.model(
            past_target,
            future_placeholder,
            past_observed_values,
            future_observed_values,
            dataset_name=dataset_name,
        )
        dec_out = x[:, :pred_len]
        if self.use_norm:
            dec_out = self.instance_norm(dec_out, "denorm")

        return dec_out  # [b l k], [b l k #patch_size]

    def loss(self, batch_data, reduce="none"):
        max_pred_len = self.prediction_length
        predict = self(
            batch_data,
            max_pred_len,
            dataset_name=None,
        )
        _, prediction_window = batch_data
        target = prediction_window

        # observed_values = batch_data.future_observed_values
        observed_values = torch.ones_like(prediction_window)

        loss = self.loss_fn(target, predict)

        loss = self.get_weighted_loss(observed_values, loss, reduce=reduce)

        if reduce == "mean":
            loss = loss.mean()
        return loss

    def forecast(self, batch_data, num_samples=None):
        # max_pred_len = batch_data.max_prediction_length if batch_data.max_prediction_length is not None else max(self.prediction_length)
        # max_pred_len = batch_data.future_target_cdf.shape[1]
        max_pred_len = self.prediction_length
        outputs = self(
            batch_data,
            max_pred_len,
            dataset_name=None,
        )
        return outputs.unsqueeze(1)

    def check_divisibility(self, pred_len, patch_size):
        if pred_len % patch_size == 0:
            return pred_len
        else:
            return (pred_len // patch_size + 1) * patch_size

    def get_weighted_loss(self, observed_values, loss, reduce="mean"):
        loss_weights, _ = observed_values.min(dim=-1, keepdim=True)
        loss = weighted_average(loss, weights=loss_weights, dim=1, reduce=reduce)
        return loss


def get_weights(sampling_weight_scheme, max_hor):
    """
    return: w [max_hor]
    """
    if sampling_weight_scheme == "random":
        i_array = np.linspace(1 + 1e-5, max_hor - 1e-3, max_hor)
        w = (1 / max_hor) * (np.log(max_hor) - np.log(i_array))
    elif sampling_weight_scheme == "const":
        w = np.array([1 / max_hor] * max_hor)
    elif sampling_weight_scheme == "none":
        return None
    else:
        raise ValueError(f"Invalid sampling scheme {sampling_weight_scheme}.")

    return torch.tensor(w)


class ElasTST(pl.LightningModule):
    def __init__(
        self,
        model: Forecaster,
        scaler: Scaler = None,
        train_pred_len_list: list = None,
        num_samples: int = 100,
        learning_rate: float = 1e-3,
        load_from_ckpt: str = None,
        sampling_weight_scheme: str = "none",
        **kwargs,
    ):
        super().__init__()
        self.num_samples = num_samples
        self.learning_rate = learning_rate
        self.load_from_ckpt = load_from_ckpt
        self.train_pred_len_list = train_pred_len_list
        self.forecaster = model

        self.scaler = IdentityScaler()

        # init the parapemetr for sampling
        self.sampling_weight_scheme = sampling_weight_scheme
        print(f"sampling_weight_scheme: {sampling_weight_scheme}")
        # self.save_hyperparameters()

    def training_forward(self, batch_data):
        # batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        # batch_data.future_target_cdf = self.scaler.transform(
        #     batch_data.future_target_cdf
        # )
        loss = self.forecaster.loss(batch_data)

        if len(loss.shape) > 1:
            loss_weights = get_weights(self.sampling_weight_scheme, loss.shape[1])
            loss = (
                loss_weights.detach().to(loss.device).unsqueeze(0).unsqueeze(-1) * loss
            ).sum(dim=1)
            loss = loss.mean()

        return loss

    def training_step(self, batch, batch_idx):
        # batch_data = ProbTSBatchData(batch, self.device)
        batch_data = batch
        loss = self.training_forward(batch_data)
        self.log("train_loss", loss, on_step=True, prog_bar=True, logger=True)
        return loss

    def evaluate(self, batch, stage="", dataloader_idx=None):
        # batch_data = ProbTSBatchData(batch, self.device)
        batch_data = batch
        pred_len = batch_data.future_target_cdf.shape[1]
        orin_past_data = batch_data.past_target_cdf[:]
        orin_future_data = batch_data.future_target_cdf[:]

        norm_past_data = self.scaler.transform(batch_data.past_target_cdf)
        norm_future_data = self.scaler.transform(batch_data.future_target_cdf)
        self.batch_size.append(orin_past_data.shape[0])

        batch_data.past_target_cdf = self.scaler.transform(batch_data.past_target_cdf)
        forecasts = self.forecaster.forecast(batch_data, self.num_samples)[
            :, :, :pred_len
        ]

        # Calculate denorm metrics
        denorm_forecasts = self.scaler.inverse_transform(forecasts)
        # TODO: calculate MSE, L1 and cross correlation metric
        return None

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        # metrics = self.evaluate(batch, stage="val", dataloader_idx=dataloader_idx)
        return 1  # TODO

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer


if __name__ == "__main__":
    model = Model(
        target_dim=96,
        context_length=96,
        prediction_length=720,
        freq="h",
        lags_list=[64],
    )

    module = ElasTST(model)
    input = torch.randn((1, 96, 1))
    output = model(input, 720)

    (output.shape)
