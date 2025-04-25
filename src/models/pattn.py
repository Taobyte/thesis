import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from src.models.utils import BaseLightningModule
from einops import rearrange


class ScaledDotProductAttention(nn.Module):
    """Scaled Dot-Product Attention"""

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):
        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention module"""

    def __init__(self, d_model=-1, n_head=8, d_k=-1, d_v=-1, dropout=0.1):
        super().__init__()
        self.n_head = n_head
        d_k = d_model // n_head
        d_v = d_k
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k**0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)  # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class Encoder_LLaTA(nn.Module):
    def __init__(self, input_dim, hidden_dim=768, num_heads=12, num_encoder_layers=1):
        super(Encoder_LLaTA, self).__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_encoder_layers
        )

    def forward(self, x):
        x = self.linear(x)
        x = self.transformer_encoder(x.transpose(0, 1)).transpose(0, 1)
        return x


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    pattn PAttn

    Decomposition-Linear
    """

    def __init__(
        self,
        look_back_window: int,
        prediction_window: int,
        patch_size: int,
        d_model: int,
    ):
        super(Model, self).__init__()
        self.seq_len = look_back_window
        self.pred_len = prediction_window
        self.patch_size = patch_size
        self.stride = patch_size // 2

        # precompute size after padding
        self.pad_size = self.stride + look_back_window

        self.d_model = d_model

        self.patch_num = max((look_back_window - self.patch_size) // self.stride + 2, 1)
        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.in_layer = nn.Linear(min(self.patch_size, self.pad_size), self.d_model)
        self.basic_attn = MultiHeadAttention(d_model=self.d_model)
        self.out_layer = nn.Linear(self.d_model * self.patch_num, prediction_window)

    def norm(self, x, dim=0, means=None, stdev=None):
        if means is not None:
            return x * stdev + means
        else:
            means = x.mean(dim, keepdim=True).detach()
            x = x - means
            stdev = torch.sqrt(
                torch.var(x, dim=dim, keepdim=True, unbiased=False) + 1e-5
            ).detach()
            x /= stdev
            return x, means, stdev

    def forward(self, x):
        x = rearrange(x, "B T C -> B C T")

        B, C = x.size(0), x.size(1)
        # [Batch, Channel, 336]
        x, means, stdev = self.norm(x, dim=2)
        # [Batch, Channel, 344]
        x = self.padding_patch_layer(x)

        # [Batch, Channel, 12, 16]
        x = x.unfold(
            dimension=-1, size=min(self.patch_size, x.shape[-1]), step=self.stride
        )

        # [Batch, Channel, 12, 768]
        x = self.in_layer(x)
        x = rearrange(x, "b c m l -> (b c) m l")
        x, _ = self.basic_attn(x, x, x)
        x = rearrange(x, "(b c) m l -> b c (m l)", b=B, c=C)
        x = self.out_layer(x)
        x = self.norm(x, means=means, stdev=stdev)
        return x


class PAttn(BaseLightningModule):
    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.02,
        lradj: str = "TST",
        tmax: int = 20,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.lradj = lradj
        self.tmax = tmax
        self.criterion = torch.nn.MSELoss()

    def model_forward(self, x):
        preds = self.model(x)
        return preds

    def _shared_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)
        loss = self.criterion(preds, prediction_window)
        return loss

    def model_specific_train_step(self, look_back_window, prediction_window):
        loss = self._shared_step(look_back_window, prediction_window)
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_step=True, on_epoch=True)
        self.log_dict({"train_loss": loss}, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        loss = self._shared_step(look_back_window, prediction_window)
        self.log_dict({"val_loss": loss}, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        epoch = self.current_epoch
        optimizer = self.trainer.optimizers[0]

        if self.lradj == "type1":
            lr_adjust = {
                epoch: self.learning_rate
                if epoch < 3
                else self.learning_rate * (0.9 ** ((epoch - 3) // 1))
            }
        elif self.lradj == "type2":
            lr_adjust = {
                epoch: self.learning_rate * (self.decay_fac ** ((epoch - 1) // 1))
            }
        elif self.lradj == "type4":
            lr_adjust = {epoch: self.learning_rate * (self.decay_fac ** ((epoch) // 1))}
        else:
            self.learning_rate = 1e-4
            lr_adjust = {
                epoch: self.learning_rate
                if epoch < 3
                else self.learning_rate * (0.9 ** ((epoch - 3) // 1))
            }
        # print("lr_adjust = {}".format(lr_adjust))

        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # print("Updating learning rate to {}".format(lr))

    def configure_optimizers(self):
        param_dict = [
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and "_proj" in n
                ],
                "lr": 1e-4,
            },
            {
                "params": [
                    p
                    for n, p in self.model.named_parameters()
                    if p.requires_grad and "_proj" not in n
                ],
                "lr": self.learning_rate,
            },
        ]

        optimizer = torch.optim.Adam([param_dict[1]], lr=self.learning_rate)

        if self.lradj == "cos":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.tmax, eta_min=1e-8
            )

            scheduler_dict = {
                "scheduler": scheduler,
                "interval": "epoch",
                "name": "CoseineAnnealingLR",
            }
            return {"optimizer": optimizer, "lr_scheduler": scheduler_dict}

        return optimizer
