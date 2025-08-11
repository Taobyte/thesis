import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


from torch.nn import Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.nn.inits import glorot, zeros
from torch_geometric.utils import degree, softmax
from torch.nn import Linear
from lightning.pytorch.core.optimizer import LightningOptimizer

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


def adjust_learning_rate(optimizer, epoch, learning_rate: float, lradj: str = "type1"):
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    elif lradj == "3":
        lr_adjust = {epoch: learning_rate if epoch < 10 else learning_rate * 0.1}
    elif lradj == "4":
        lr_adjust = {epoch: learning_rate if epoch < 15 else learning_rate * 0.1}
    elif lradj == "5":
        lr_adjust = {epoch: learning_rate if epoch < 25 else learning_rate * 0.1}
    elif lradj == "6":
        lr_adjust = {epoch: learning_rate if epoch < 5 else learning_rate * 0.1}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(
        self, temperature, attn_dropout=0.2
    ):  ##temperature=11.3137084对于128维度进行了根号d
        super().__init__()

        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before
        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(
            d_model, n_head * d_k, bias=False
        )  ##[input 512,output 768]
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        nn.init.xavier_uniform_(self.w_qs.weight)
        nn.init.xavier_uniform_(self.w_ks.weight)
        nn.init.xavier_uniform_(self.w_vs.weight)

        self.fc = nn.Linear(d_v * n_head, d_model)
        nn.init.xavier_uniform_(self.fc.weight)

        self.attention = ScaledDotProductAttention(
            temperature=d_k**0.5, attn_dropout=dropout
        )

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head  ###d_k,d_v=128 n_head=6
        sz_b, len_q, len_k, len_v = (
            q.size(0),
            q.size(1),
            k.size(1),
            v.size(1),
        )  ###sz_b=32, len_q=223, len_k=223, len_v=223

        residual = q  # 32 223 512
        if self.normalize_before:
            q = self.layer_norm(q)

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)  # [32,223,6,128]
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)  # [32,223,6,128]
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)  # [32,223,6,128]

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  ###q,k,v=#[32,6,223,128]

        if mask is not None:  ###mask[32,223,223]
            if len(mask.size()) == 3:
                mask = mask.unsqueeze(
                    1
                )  # For head axis broadcasting.###[32,1,223,223]为了和q,k,v维度对齐

        output, attn = self.attention(
            q, k, v, mask=mask
        )  # attn=[32,6,223,223]    output=[32,6,223,128]
        # print(output.shape)   #[32,6,223,128]

        # Transpose to move the head dimension back: b x lq x n x dv  32x223x6x128
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        output = (
            output.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        )  ###[32,223,6x128]
        output = self.dropout(self.fc(output))  ###[32,223,512]
        output += residual  ###[32,223,512]

        if not self.normalize_before:
            output = self.layer_norm(output)
        return output, attn


class PositionwiseFeedForward(nn.Module):
    """Two-layer position-wise feed-forward neural network."""

    def __init__(self, d_in, d_hid, dropout=0.1, normalize_before=True):
        super().__init__()

        self.normalize_before = normalize_before

        self.w_1 = nn.Linear(d_in, d_hid)
        self.w_2 = nn.Linear(d_hid, d_in)

        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.layer_norm = GraphNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        if self.normalize_before:
            x = self.layer_norm(x)

        x = F.gelu(self.w_1(x))
        x = self.dropout(x)
        x = self.w_2(x)
        x = self.dropout(x)
        x = x + residual

        if not self.normalize_before:
            x = self.layer_norm(x)
        return x


####通用的
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


####通用的
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
        )
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )

    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x


####通用的
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


####通用的
class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model):
        super(TimeFeatureEmbedding, self).__init__()

        d_inp = 4
        self.embed = nn.Linear(d_inp, d_model)

    def forward(self, x):
        return self.embed(x)


"""Embedding modules. The DataEmbedding is used by the ETT dataset for long range forecasting."""


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = TemporalEmbedding(d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        if x_mark is None:
            x = self.value_embedding(x) + self.position_embedding(x)
        else:
            # b=self.value_embedding(x)
            # c=self.temporal_embedding(x_mark)
            # d=self.position_embedding(x)
            x = (
                self.value_embedding(x)
                + self.position_embedding(x)
                + self.temporal_embedding(x_mark)
            )
        return self.dropout(x)


# new_added
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


class DataEmbedding_new(nn.Module):
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
        x = (
            self.value_embedding(x)
            + self.temporal_embedding(x_mark)
            + self.position_embedding(x)
        )
        # x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


"""The CustomEmbedding is used by the electricity dataset and app flow dataset for long range forecasting."""


class CustomEmbedding(nn.Module):
    def __init__(self, c_in, d_model, temporal_size, seq_num, dropout=0.1):
        super(CustomEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.temporal_embedding = nn.Linear(temporal_size, d_model)
        self.seqid_embedding = nn.Embedding(seq_num, d_model)

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        #######origional
        x = (
            self.value_embedding(x)
            + self.position_embedding(x)
            + self.temporal_embedding(x_mark[:, :, :-1])
            + self.seqid_embedding(x_mark[:, :, -1].long())
        )

        #####change by ourself
        # x = self.value_embedding(x) + self.position_embedding(x) + self.temporal_embedding(x_mark[:, :, :-1])
        # x = self.value_embedding(x) + self.position_embedding(x)

        return self.dropout(x)


"""The SingleStepEmbedding is used by all datasets for single step forecasting."""


class SingleStepEmbedding(nn.Module):
    def __init__(self, cov_size, num_seq, d_model, input_size, device):
        super().__init__()

        self.cov_size = cov_size
        self.num_class = num_seq
        self.cov_emb = nn.Linear(cov_size + 1, d_model)
        padding = 1 if torch.__version__ >= "1.5.0" else 2
        self.data_emb = nn.Conv1d(
            in_channels=1,
            out_channels=d_model,
            kernel_size=3,
            padding=padding,
            padding_mode="circular",
        )

        self.position = torch.arange(input_size, device=device).unsqueeze(0)
        self.position_vec = torch.tensor(
            [math.pow(10000.0, 2.0 * (i // 2) / d_model) for i in range(d_model)],
            device=device,
        )

        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_in", nonlinearity="leaky_relu"
                )
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def transformer_embedding(self, position, vector):
        """
        Input: batch*seq_len.
        Output: batch*seq_len*d_model.
        """
        result = position.unsqueeze(-1) / vector
        result[:, :, 0::2] = torch.sin(result[:, :, 0::2])
        result[:, :, 1::2] = torch.cos(result[:, :, 1::2])
        return result

    def forward(self, x):
        covs = x[:, :, 1 : (1 + self.cov_size)]
        seq_ids = ((x[:, :, -1] / self.num_class) - 0.5).unsqueeze(2)
        covs = torch.cat([covs, seq_ids], dim=-1)
        cov_embedding = self.cov_emb(covs)
        data_embedding = self.data_emb(
            x[:, :, 0].unsqueeze(2).permute(0, 2, 1)
        ).transpose(1, 2)
        embedding = cov_embedding + data_embedding

        position = self.position.repeat(len(x), 1).to(x.device)
        position_emb = self.transformer_embedding(
            position, self.position_vec.to(x.device)
        )

        embedding += position_emb

        return embedding


def refer_points(all_sizes, window_size, device):
    """Gather features from PAM's pyramid sequences"""
    input_size = all_sizes[0]  ###all_size=[169,42,10,2]
    indexes = torch.zeros(input_size, len(all_sizes), device=device)  ###index=[169,4]

    for i in range(input_size):
        indexes[i][0] = i
        former_index = i
        for j in range(1, len(all_sizes)):
            start = sum(all_sizes[:j])
            inner_layer_idx = former_index - (start - all_sizes[j - 1])
            former_index = start + min(
                inner_layer_idx // window_size[j - 1], all_sizes[j] - 1
            )
            indexes[i][j] = former_index

    indexes = indexes.unsqueeze(0).unsqueeze(3)  ###unsqueeze(0)在0这个维度增加一维

    return indexes.long()


def get_subsequent_mask(input_size, window_size, predict_step, truncate):
    """Get causal attention mask for decoder."""
    if truncate:
        mask = torch.zeros(predict_step, input_size + predict_step)
        for i in range(predict_step):
            mask[i][: input_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)
    else:
        all_size = []
        all_size.append(input_size)
        for i in range(len(window_size)):
            layer_size = math.floor(all_size[i] / window_size[i])
            all_size.append(layer_size)
        all_size = sum(all_size)
        mask = torch.zeros(predict_step, all_size + predict_step)
        for i in range(predict_step):
            mask[i][: all_size + i + 1] = 1
        mask = (1 - mask).bool().unsqueeze(0)

    return mask


def get_q_k(input_size, window_size, stride, device):
    """
    Get the index of the key that a given query needs to attend to.
    """
    second_length = input_size // stride
    second_last = input_size - (second_length - 1) * stride
    third_start = input_size + second_length
    third_length = second_length // stride
    third_last = second_length - (third_length - 1) * stride
    max_attn = max(second_last, third_last)
    fourth_start = third_start + third_length
    fourth_length = third_length // stride
    full_length = fourth_start + fourth_length
    fourth_last = third_length - (fourth_length - 1) * stride
    max_attn = max(third_last, fourth_last)

    max_attn += window_size + 1
    mask = torch.zeros(full_length, max_attn, dtype=torch.int32, device=device) - 1

    for i in range(input_size):
        mask[i, 0:window_size] = i + torch.arange(window_size) - window_size // 2
        mask[i, mask[i] > input_size - 1] = -1

        mask[i, -1] = i // stride + input_size
        mask[i][mask[i] > third_start - 1] = third_start - 1
    for i in range(second_length):
        mask[input_size + i, 0:window_size] = (
            input_size + i + torch.arange(window_size) - window_size // 2
        )
        mask[input_size + i, mask[input_size + i] < input_size] = -1
        mask[input_size + i, mask[input_size + i] > third_start - 1] = -1

        if i < second_length - 1:
            mask[input_size + i, window_size : (window_size + stride)] = (
                torch.arange(stride) + i * stride
            )
        else:
            mask[input_size + i, window_size : (window_size + second_last)] = (
                torch.arange(second_last) + i * stride
            )

        mask[input_size + i, -1] = i // stride + third_start
        mask[input_size + i, mask[input_size + i] > fourth_start - 1] = fourth_start - 1
    for i in range(third_length):
        mask[third_start + i, 0:window_size] = (
            third_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[third_start + i, mask[third_start + i] < third_start] = -1
        mask[third_start + i, mask[third_start + i] > fourth_start - 1] = -1

        if i < third_length - 1:
            mask[third_start + i, window_size : (window_size + stride)] = (
                input_size + torch.arange(stride) + i * stride
            )
        else:
            mask[third_start + i, window_size : (window_size + third_last)] = (
                input_size + torch.arange(third_last) + i * stride
            )

        mask[third_start + i, -1] = i // stride + fourth_start
        mask[third_start + i, mask[third_start + i] > full_length - 1] = full_length - 1
    for i in range(fourth_length):
        mask[fourth_start + i, 0:window_size] = (
            fourth_start + i + torch.arange(window_size) - window_size // 2
        )
        mask[fourth_start + i, mask[fourth_start + i] < fourth_start] = -1
        mask[fourth_start + i, mask[fourth_start + i] > full_length - 1] = -1

        if i < fourth_length - 1:
            mask[fourth_start + i, window_size : (window_size + stride)] = (
                third_start + torch.arange(stride) + i * stride
            )
        else:
            mask[fourth_start + i, window_size : (window_size + fourth_last)] = (
                third_start + torch.arange(fourth_last) + i * stride
            )

    return mask


def get_k_q(q_k_mask):
    """
    Get the index of the query that can attend to the given key.
    """
    k_q_mask = q_k_mask.clone()
    for i in range(len(q_k_mask)):
        for j in range(len(q_k_mask[0])):
            if q_k_mask[i, j] >= 0:
                k_q_mask[i, j] = torch.where(q_k_mask[q_k_mask[i, j]] == i)[0]

    return k_q_mask


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
        normalize_before=True,
        use_tvm=False,
        q_k_mask=None,
        k_q_mask=None,
    ):
        super(EncoderLayer, self).__init__()
        self.use_tvm = use_tvm
        if use_tvm:
            from .PAM_TVM import PyramidalAttention

            self.slf_attn = PyramidalAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                normalize_before=normalize_before,
                q_k_mask=q_k_mask,
                k_q_mask=k_q_mask,
            )
            ###n_head=6, d_model=512, d_k=128, d_v=128
        else:
            self.slf_attn = MultiHeadAttention(
                n_head,
                d_model,
                d_k,
                d_v,
                dropout=dropout,
                normalize_before=normalize_before,
            )

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, enc_input, slf_attn_mask=None):
        if self.use_tvm:
            enc_output = self.slf_attn(enc_input)
            enc_slf_attn = None
        else:
            enc_output, enc_slf_attn = self.slf_attn(
                enc_input, enc_input, enc_input, mask=slf_attn_mask
            )  ##enc_input[32,223,512] mask[32,223,223]

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class DecoderLayer(nn.Module):
    """Compose with two layers"""

    def __init__(
        self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1, normalize_before=True
    ):
        super(DecoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head,
            d_model,
            d_k,
            d_v,
            dropout=dropout,
            normalize_before=normalize_before,
        )
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout, normalize_before=normalize_before
        )

    def forward(self, Q, K, V, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(Q, K, V, mask=slf_attn_mask)

        enc_output = self.pos_ffn(enc_output)

        return enc_output, enc_slf_attn


class ConvLayer(nn.Module):
    def __init__(self, c_in, window_size):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(
            in_channels=c_in,
            out_channels=c_in,
            kernel_size=window_size,
            stride=window_size,
        )
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()

    def forward(self, x):
        x = self.downConv(x)  ####input_x[32,128,42]
        x = self.norm(x)
        x = self.activation(x)
        return x


class Conv_Constructnew(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        # if not isinstance(window_size, list):
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size),
        #         ConvLayer(d_model, window_size)
        #         ])
        # else:
        #     self.conv_layers = nn.ModuleList([
        #         ConvLayer(d_model, window_size[0]),
        #         ConvLayer(d_model, window_size[1]),
        #         ConvLayer(d_model, window_size[2])
        #         ])
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                    # ConvLayer(d_model, window_size)
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size[0]),
                    ConvLayer(d_model, window_size[1]),
                    # ConvLayer(d_model, window_size[2])
                ]
            )

        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Conv_Construct(nn.Module):
    """Convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Conv_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                    ConvLayer(d_model, window_size),
                ]
            )
        else:
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_model, window_size[0]),
                    ConvLayer(d_model, window_size[1]),
                    ConvLayer(d_model, window_size[2]),
                ]
            )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.permute(0, 2, 1)
        all_inputs.append(enc_input)

        for i in range(len(self.conv_layers)):
            enc_input = self.conv_layers[i](enc_input)
            all_inputs.append(enc_input)

        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        all_inputs = self.norm(all_inputs)

        return all_inputs


class Bottleneck_Construct(nn.Module):
    """Bottleneck convolution CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(Bottleneck_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.conv_layers = nn.ModuleList(
                [
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                    ConvLayer(d_inner, window_size),
                ]
            )
        else:
            self.conv_layers = []
            for i in range(len(window_size)):
                self.conv_layers.append(ConvLayer(d_inner, window_size[i]))
            self.conv_layers = nn.ModuleList(self.conv_layers)

        self.up = Linear(d_inner, d_model)  ####d_inner128 d_model=512
        self.down = Linear(d_model, d_inner)  ####d_model=512 d_inner128
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):  ####[32,169,512]
        temp_input = self.down(enc_input).permute(
            0, 2, 1
        )  ####先下采样，变为[32,169,128],再交换第1维度和第二维度-->[32,128,169]
        all_inputs = []
        all_inputs.append(temp_input.permute(0, 2, 1))
        ####对169个节点进行卷积
        for i in range(len(self.conv_layers)):
            temp_input = self.conv_layers[i](
                temp_input
            )  ####第一次[32,128,42]第二次[32,128,10]第三次[32,128,2]
            all_inputs.append(temp_input.permute(0, 2, 1))
        """
        # print(all_inputs[1].shape())
        # hour_trend=self.up(all_inputs[0].transpose(1, 2))
        # day_trend=self.up(all_inputs[1].transpose(1, 2))
        # week_trend=self.up(all_inputs[2].transpose(1, 2))
        all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)####[32,54,128]
        all_inputs = self.up(all_inputs)####[32,54,512]
        all_inputs = torch.cat([enc_input, all_inputs], dim=1)####enc_input[32,169,512]all_input=[32,223,512]

        all_inputs = self.norm(all_inputs)####[32,223,512]
        # hour_trend = self.norm(hour_trend)
        # day_trend = self.norm(day_trend)
        # week_trend = self.norm(week_trend)
        """
        return all_inputs


class MaxPooling_Construct(nn.Module):
    """Max pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(MaxPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [
                    nn.MaxPool1d(kernel_size=window_size),
                    nn.MaxPool1d(kernel_size=window_size),
                    nn.MaxPool1d(kernel_size=window_size),
                ]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [
                    nn.MaxPool1d(kernel_size=window_size[0]),
                    nn.MaxPool1d(kernel_size=window_size[1]),
                ]
            )
            # self.pooling_layers = nn.ModuleList([
            #     nn.MaxPool1d(kernel_size=window_size[0]),
            #     nn.MaxPool1d(kernel_size=window_size[1]),
            #     nn.MaxPool1d(kernel_size=window_size[2])
            #     ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        # all_inputs.append(enc_input)
        all_inputs.append(enc_input.transpose(1, 2))

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            # all_inputs.append(enc_input)
            all_inputs.append(enc_input.transpose(1, 2))

        # all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        # all_inputs = self.norm(all_inputs)

        return all_inputs


class AvgPooling_Construct(nn.Module):
    """Average pooling CSCM"""

    def __init__(self, d_model, window_size, d_inner):
        super(AvgPooling_Construct, self).__init__()
        if not isinstance(window_size, list):
            self.pooling_layers = nn.ModuleList(
                [
                    nn.AvgPool1d(kernel_size=window_size),
                    nn.AvgPool1d(kernel_size=window_size),
                    nn.AvgPool1d(kernel_size=window_size),
                ]
            )
        else:
            self.pooling_layers = nn.ModuleList(
                [
                    nn.MaxPool1d(kernel_size=window_size[0]),
                    nn.MaxPool1d(kernel_size=window_size[1]),
                ]
            )
            # self.pooling_layers = nn.ModuleList([
            #     nn.AvgPool1d(kernel_size=window_size[0]),
            #     nn.AvgPool1d(kernel_size=window_size[1]),
            #     nn.AvgPool1d(kernel_size=window_size[2])
            #     ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, enc_input):
        all_inputs = []
        enc_input = enc_input.transpose(1, 2).contiguous()
        # all_inputs.append(enc_input)
        all_inputs.append(enc_input.transpose(1, 2))

        for layer in self.pooling_layers:
            enc_input = layer(enc_input)
            # all_inputs.append(enc_input)
            all_inputs.append(enc_input.transpose(1, 2))

        # all_inputs = torch.cat(all_inputs, dim=2).transpose(1, 2)
        # all_inputs = self.norm(all_inputs)

        return all_inputs


class Predictor(nn.Module):
    def __init__(self, dim, num_types):
        super().__init__()

        self.linear = nn.Linear(dim, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data):
        out = self.linear(data)
        out = out
        return out


class Decoder(nn.Module):
    """A encoder model with self attention mechanism."""

    def __init__(self, opt, mask):
        super().__init__()

        self.model_type = opt.model
        self.mask = mask

        self.layers = nn.ModuleList(
            [
                DecoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    opt.d_k,
                    opt.d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                ),
                DecoderLayer(
                    opt.d_model,
                    opt.d_inner_hid,
                    opt.n_head,
                    opt.d_k,
                    opt.d_v,
                    dropout=opt.dropout,
                    normalize_before=False,
                ),
            ]
        )

        if opt.embed_type == "CustomEmbedding":
            self.dec_embedding = CustomEmbedding(
                opt.enc_in, opt.d_model, opt.covariate_size, opt.seq_num, opt.dropout
            )
        else:
            self.dec_embedding = DataEmbedding(opt.enc_in, opt.d_model, opt.dropout)

    def forward(self, x_dec, x_mark_dec, refer):
        dec_enc = self.dec_embedding(x_dec, x_mark_dec)

        dec_enc, _ = self.layers[0](dec_enc, refer, refer)
        refer_enc = torch.cat([refer, dec_enc], dim=1)
        mask = self.mask.repeat(len(dec_enc), 1, 1).to(dec_enc.device)
        dec_enc, _ = self.layers[1](dec_enc, refer_enc, refer_enc, slf_attn_mask=mask)

        return dec_enc


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 96,
        channels: int = 1,
        individual: bool = False,
        CSCM: str = "Bottleneck_Construct",
        d_model: int = 512,
        hn1: int = 50,
        hn2: int = 20,
        hn3: int = 10,
        window_size: list[int] = [4, 4],
        k: int = 3,  # eta parameter from the paper
        beta: float = 0.5,
        gamma: float = 4.2,
        inner_size: int = 5,
        use_norm: bool = True,
    ):
        super(Model, self).__init__()
        hyper_num = [hn1, hn2, hn3]  # number of hyperedges at scales 1, 2, 3
        window_size = list(window_size)  # aggregation window at scale 1, 2
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.individual = individual
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Tran = nn.Linear(self.pred_len, self.pred_len)

        self.all_size = get_mask(seq_len, window_size)
        self.Ms_length = sum(self.all_size)
        self.conv_layers = eval(CSCM)(channels, window_size, channels)
        self.out_tran = nn.Linear(self.Ms_length, self.pred_len)
        self.out_tran.weight = nn.Parameter(
            (1 / self.Ms_length) * torch.ones([self.pred_len, self.Ms_length])
        )
        self.chan_tran = nn.Linear(d_model, channels)
        self.inter_tran = nn.Linear(80, self.pred_len)
        self.concat_tra = nn.Linear(320, self.pred_len)

        self.alpha = 3
        self.k = k

        self.window_size = window_size
        self.multiadphyper = multi_adaptive_hypergraoh(
            d_model=d_model,
            hyper_num=hyper_num,
            inner_size=inner_size,
            k=k,
            seq_len=seq_len,
            window_size=window_size,
            beta=beta,
        )
        self.hyper_num1 = hyper_num
        self.hyconv = nn.ModuleList()
        self.hyperedge_atten = SelfAttentionLayer(channels)
        for i in range(len(self.hyper_num1)):
            self.hyconv.append(HypergraphConv(channels, channels, gamma=gamma))

        self.weight = nn.Parameter(torch.randn(self.pred_len, 76))

        self.argg = nn.ModuleList()
        for i in range(len(self.hyper_num1)):
            self.argg.append(nn.Linear(self.all_size[i], self.pred_len))
        self.chan_tran = nn.Linear(channels, channels)

        self.use_norm = use_norm

    def forward(self, x):
        # normalization
        if self.use_norm:
            mean_enc = x.mean(1, keepdim=True).detach()
            x = x - mean_enc
            std_enc = torch.sqrt(
                torch.var(x, dim=1, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x = x / std_enc

        adj_matrix = self.multiadphyper(x)
        seq_enc = self.conv_layers(x)

        sum_hyper_list = []
        for i in range(len(self.hyper_num1)):
            mask = torch.tensor(adj_matrix[i]).to(x.device)
            ###inter-scale
            node_value = seq_enc[i].permute(0, 2, 1)
            node_value = torch.tensor(node_value).to(x.device)
            edge_sums = {}
            for edge_id, node_id in zip(mask[1], mask[0]):
                if edge_id not in edge_sums:
                    edge_id = edge_id.item()
                    node_id = node_id.item()
                    edge_sums[edge_id] = node_value[:, :, node_id]
                else:
                    edge_sums[edge_id] += node_value[:, :, node_id]

            for edge_id, sum_value in edge_sums.items():
                sum_value = sum_value.unsqueeze(1)
                sum_hyper_list.append(sum_value)

            ###intra-scale
            output, constrainloss = self.hyconv[i](seq_enc[i], mask)

            if i == 0:
                result_tensor = output
                result_conloss = constrainloss
            else:
                result_tensor = torch.cat((result_tensor, output), dim=1)
                result_conloss += constrainloss

        sum_hyper_list = torch.cat(sum_hyper_list, dim=1)
        sum_hyper_list = sum_hyper_list.to(x.device)
        padding_need = 80 - sum_hyper_list.size(1)
        hyperedge_attention = self.hyperedge_atten(sum_hyper_list)
        pad = torch.nn.functional.pad(
            hyperedge_attention, (0, 0, 0, padding_need, 0, 0)
        )

        if self.individual:
            output = torch.zeros(
                [x.size(0), self.pred_len, x.size(2)], dtype=x.dtype
            ).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1))

            x_out = self.out_tran(result_tensor.permute(0, 2, 1))  ###ori
            x_out_inter = self.inter_tran(pad.permute(0, 2, 1))

        x = x_out + x + x_out_inter
        x = self.Linear_Tran(x).permute(0, 2, 1)

        if self.use_norm:
            x = x * std_enc + mean_enc

        return x, result_conloss  # [Batch, Output length, Channel]


class HypergraphConv(MessagePassing):
    def __init__(
        self,
        in_channels,
        out_channels,
        use_attention=True,
        heads=1,
        concat=True,
        negative_slope=0.2,
        dropout=0.1,
        bias=False,
        gamma: float = 4.2,
    ):
        super(HypergraphConv, self).__init__(aggr="add", node_dim=0)
        self.soft = nn.Softmax(dim=0)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.use_attention = use_attention

        self.gamma = gamma  # gamma parameter from the paper used in the hyperedge loss

        if self.use_attention:
            self.heads = heads
            self.concat = concat
            self.negative_slope = negative_slope
            self.dropout = dropout
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

            self.att = Parameter(torch.Tensor(1, heads, 2 * int(out_channels / heads)))

        else:
            self.heads = 1
            self.concat = True
            self.weight = Parameter(torch.Tensor(in_channels, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.weight)
        if self.use_attention:
            glorot(self.att)
        zeros(self.bias)

    def __forward__(self, x, hyperedge_index, alpha=None):
        D = degree(hyperedge_index[0], x.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)
        # --------------------------------------------------------
        B[B == float("inf")] = 0

        self.flow = "source_to_target"
        out = self.propagate(hyperedge_index, x=x, norm=B, alpha=alpha)
        self.flow = "target_to_source"
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)

        return out

    def message(self, x_j, edge_index_i, norm, alpha):
        out = norm[edge_index_i].view(-1, 1, 1) * x_j
        if alpha is not None:
            out = alpha.unsqueeze(-1) * out
        return out

    def forward(self, x, hyperedge_index):
        x = torch.matmul(x, self.weight)
        x1 = x.transpose(0, 1)
        x_i = torch.index_select(x1, dim=0, index=hyperedge_index[0])
        edge_sums = {}

        for edge_id, node_id in zip(hyperedge_index[1], hyperedge_index[0]):
            if edge_id not in edge_sums:
                edge_id = edge_id.item()
                node_id = node_id.item()
                edge_sums[edge_id] = x1[node_id, :, :]
            else:
                edge_sums[edge_id] += x1[node_id, :, :]
        result_list = torch.stack([value for value in edge_sums.values()], dim=0)
        x_j = torch.index_select(result_list, dim=0, index=hyperedge_index[1])
        loss_hyper = 0
        for k in range(len(edge_sums)):
            for m in range(len(edge_sums)):
                inner_product = torch.sum(
                    edge_sums[k] * edge_sums[m], dim=1, keepdim=True
                )
                norm_q_i = torch.norm(edge_sums[k], dim=1, keepdim=True)
                norm_q_j = torch.norm(edge_sums[m], dim=1, keepdim=True)
                alpha = inner_product / (norm_q_i * norm_q_j)
                distan = torch.norm(edge_sums[k] - edge_sums[m], dim=1, keepdim=True)
                loss_item = alpha * distan + (1 - alpha) * (
                    torch.clamp(torch.tensor(self.gamma) - distan, min=0.0)
                )
                loss_hyper += torch.abs(torch.mean(loss_item))

        loss_hyper = loss_hyper / ((len(edge_sums) + 1) ** 2)
        alpha = (torch.cat([x_i, x_j], dim=-1) * self.att).sum(dim=-1)
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, hyperedge_index[0], num_nodes=x1.size(0))
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        D = degree(hyperedge_index[0], x1.size(0), x.dtype)
        num_edges = 2 * (hyperedge_index[1].max().item() + 1)
        B = 1.0 / degree(hyperedge_index[1], int(num_edges / 2), x.dtype)
        B[B == float("inf")] = 0
        self.flow = "source_to_target"
        out = self.propagate(hyperedge_index, x=x1, norm=B, alpha=alpha)
        self.flow = "target_to_source"
        out = self.propagate(hyperedge_index, x=out, norm=D, alpha=alpha)
        out = out.transpose(0, 1)
        constrain_loss = x_i - x_j
        constrain_lossfin1 = torch.mean(constrain_loss)
        constrain_losstotal = abs(constrain_lossfin1) + loss_hyper
        return out, constrain_losstotal

    def __repr__(self):
        return "{}({}, {})".format(
            self.__class__.__name__, self.in_channels, self.out_channels
        )


class multi_adaptive_hypergraoh(nn.Module):
    def __init__(
        self,
        seq_len: int,
        window_size: list[int],
        inner_size: int,
        d_model: int,
        hyper_num: int,
        k: int,
        beta: float = 0.5,
    ):
        super(multi_adaptive_hypergraoh, self).__init__()
        self.seq_len = seq_len
        self.window_size = window_size
        self.inner_size = inner_size
        self.dim = d_model
        self.hyper_num = hyper_num
        self.alpha = 3
        self.beta = beta
        self.k = k
        self.embedhy = nn.ModuleList()
        self.embednod = nn.ModuleList()
        self.linhy = nn.ModuleList()
        self.linnod = nn.ModuleList()
        for i in range(len(self.hyper_num)):
            self.embedhy.append(nn.Embedding(self.hyper_num[i], self.dim))
            self.linhy.append(nn.Linear(self.dim, self.dim))
            self.linnod.append(nn.Linear(self.dim, self.dim))
            if i == 0:
                self.embednod.append(nn.Embedding(self.seq_len, self.dim))
            else:
                product = math.prod(self.window_size[:i])
                layer_size = max(math.floor(self.seq_len / product), 1)
                self.embednod.append(nn.Embedding(int(layer_size), self.dim))

        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        node_num = []
        node_num.append(self.seq_len)
        for i in range(len(self.window_size)):
            layer_size = max(
                math.floor(node_num[i] / self.window_size[i]), 1
            )  # added min that we have at least value 1 and not 0
            node_num.append(layer_size)
        hyperedge_all = []

        for i in range(len(self.hyper_num)):
            # sequence of integers of size hyper_num[i] and node_num[i]
            hypidxc = torch.arange(self.hyper_num[i]).to(x.device)
            nodeidx = torch.arange(node_num[i]).to(x.device)

            # hyperedge and node embeddings
            hyperen = self.embedhy[i](hypidxc)
            nodeec = self.embednod[i](nodeidx)

            a = torch.mm(nodeec, hyperen.transpose(1, 0))
            adj = F.softmax(F.relu(self.alpha * a))
            mask = torch.zeros(nodeec.size(0), hyperen.size(0)).to(x.device)
            mask.fill_(float("0"))
            s1, t1 = adj.topk(min(adj.size(1), self.k), 1)
            mask.scatter_(1, t1, s1.fill_(1))
            adj = adj * mask
            adj = torch.where(
                adj > self.beta,
                torch.tensor(1).to(x.device),
                torch.tensor(0).to(x.device),
            )
            adj = adj[:, (adj != 0).any(dim=0)]
            matrix_array = adj.clone().detach().to(dtype=torch.int)
            result_list = [
                list(torch.nonzero(matrix_array[:, col]).flatten().tolist())
                for col in range(matrix_array.shape[1])
            ]

            node_list = torch.cat(
                [torch.tensor(sublist) for sublist in result_list if len(sublist) > 0]
            ).tolist()
            count_list = list(torch.sum(adj, dim=0).tolist())
            hperedge_list = torch.cat(
                [
                    torch.full((count,), idx)
                    for idx, count in enumerate(count_list, start=0)
                ]
            ).tolist()
            hypergraph = np.vstack((node_list, hperedge_list))
            hyperedge_all.append(hypergraph)

        return hyperedge_all


class SelfAttentionLayer(nn.Module):
    def __init__(self, enc_in: int):
        super(SelfAttentionLayer, self).__init__()
        self.query_weight = nn.Linear(enc_in, enc_in)
        self.key_weight = nn.Linear(enc_in, enc_in)
        self.value_weight = nn.Linear(enc_in, enc_in)

    def forward(self, x):
        q = self.query_weight(x)
        k = self.key_weight(x)
        v = self.value_weight(x)
        attention_scores = F.softmax(
            torch.matmul(q, k.transpose(1, 2)) / (k.shape[-1] ** 0.5), dim=-1
        )
        attended_values = torch.matmul(attention_scores, v)

        return attended_values


def get_mask(input_size, window_size):
    """Get the attention mask of HyperGraphConv"""
    # Get the size of all layers
    # window_size=[4,4,4]
    all_size = []
    all_size.append(input_size)
    for i in range(len(window_size)):
        layer_size = math.floor(all_size[i] / window_size[i])
        all_size.append(layer_size)
    return all_size


class AdaMSHyper(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        loss: str = "MSE",
        lradj: str = "type1",
        use_amp: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model

        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate
        self.lradj = lradj
        self.automatic_optimization = False

    def model_forward(self, look_back_window):
        preds, _ = self.model(look_back_window)
        return preds

    def model_specific_train_step(self, look_back_window, prediction_window):
        # opt_1, opt_2 = self.configure_optimizers()

        opt_1, opt_2 = self.optimizers(use_pl_optimizer=True)

        opt_1.zero_grad()
        opt_2.zero_grad()
        preds, constraint_loss = self.model(look_back_window)

        preds = preds[:, :, : prediction_window.shape[-1]]

        assert preds.shape == prediction_window.shape

        mse_loss = self.criterion(preds, prediction_window)
        constraint_loss = constraint_loss.abs()
        self.manual_backward(mse_loss, retain_graph=True)
        self.manual_backward(constraint_loss)
        opt_1.step()
        opt_2.step()
        self.log("train_loss", mse_loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(
            {
                "constraint_loss": constraint_loss,
                "total_loss": mse_loss + constraint_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return mse_loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds, constraint_loss = self.model(look_back_window)
        preds = preds[:, :, : prediction_window.shape[-1]]
        mse_loss = self.criterion(preds, prediction_window)
        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            mse_loss = mae_criterion(preds, prediction_window)
        self.log("val_loss", mse_loss, on_step=True, on_epoch=True, logger=True)
        self.log_dict(
            {
                "constraint_loss": constraint_loss,
                "total_loss": mse_loss + constraint_loss,
            },
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return mse_loss

    def on_train_epoch_start(self):
        # Adjust learning rate at the start of each epoch
        adjust_learning_rate(
            self.trainer.optimizers[0],
            self.current_epoch + 1,
            self.learning_rate,
            self.lradj,
        )
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer_1 = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        optimizer_2 = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        if self.automatic_optimization is False:
            strategy = self.trainer.strategy
            optimizer_1 = LightningOptimizer._to_lightning_optimizer(
                optimizer_1, strategy
            )
            optimizer_2 = LightningOptimizer._to_lightning_optimizer(
                optimizer_2, strategy
            )

        return [optimizer_1, optimizer_2]
