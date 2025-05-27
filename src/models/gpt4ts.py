import math
import torch
import torch.nn as nn

from transformers.models.gpt2.modeling_gpt2 import GPT2Model

from src.models.utils import BaseLightningModule
from src.losses import get_loss_fn


def adjust_learning_rate(
    optimizer: torch.optim.Optimizer, epoch: int, lradj: str, learning_rate: float
):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if lradj == "type1":
        lr_adjust = {epoch: learning_rate * (0.5 ** ((epoch - 1) // 1))}
    elif lradj == "type7":
        lr_adjust = {epoch: learning_rate * (0.7 ** ((epoch - 1) // 1))}
    elif lradj == "type6":
        lr_adjust = {epoch: learning_rate * (0.6 ** ((epoch - 1) // 1))}
    elif lradj == "type2":
        lr_adjust = {2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6, 10: 5e-7, 15: 1e-7, 20: 5e-8}
    else:
        lr_adjust = {epoch: learning_rate}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        print("Updating learning rate to {}".format(lr))


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


class DataEmbedding_wo_pos(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_pos, self).__init__()

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
            x = self.value_embedding(x)
        else:
            x = self.value_embedding(x) + self.temporal_embedding(x_mark)
        return self.dropout(x)


class PatchEmbedding(nn.Module):
    def __init__(self, d_model, patch_len, stride, dropout):
        super(PatchEmbedding, self).__init__()
        # Patching
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch_layer = nn.ReplicationPad1d((0, stride))

        # Backbone, Input encoding: projection of feature vectors onto a d-dim vector space
        self.value_embedding = TokenEmbedding(patch_len, d_model)

        # Positional embedding
        self.position_embedding = PositionalEmbedding(d_model)

        # Residual dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # do patching
        n_vars = x.shape[1]
        x = self.padding_patch_layer(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        # Input encoding
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x), n_vars


class DataEmbedding_wo_time(nn.Module):
    def __init__(self, c_in, d_model, embed_type="fixed", freq="h", dropout=0.1):
        super(DataEmbedding_wo_time, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class Model(nn.Module):
    def __init__(
        self,
        ln: int = 0,
        pred_len: int = 96,
        seq_len: int = 96,
        patch_size: int = 1,
        stride: int = 1,
        d_ff: int = 128,
        enc_in: int = 1,
        c_out: int = 1,
        d_model: int = 128,
        embed: str = "timeF",
        freq: str = "m",
        dropout: float = 0.1,
        gpt_layers: int = 6,
        mlp: str = 0,
        use_gpu: bool = False,
    ):
        super(Model, self).__init__()
        self.is_ln = ln
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.seq_len = seq_len
        self.d_ff = d_ff
        self.patch_num = (seq_len + self.pred_len - self.patch_size) // self.stride + 1

        self.padding_patch_layer = nn.ReplicationPad1d((0, self.stride))
        self.patch_num += 1
        self.enc_embedding = DataEmbedding(
            enc_in * self.patch_size,
            d_model,
            embed,
            freq,
            dropout,
        )

        self.gpt2 = GPT2Model.from_pretrained(
            "gpt2", output_attentions=True, output_hidden_states=True
        )
        self.gpt2.h = self.gpt2.h[:gpt_layers]

        for i, (name, param) in enumerate(self.gpt2.named_parameters()):
            if "ln" in name or "wpe" in name:  # or 'mlp' in name:
                param.requires_grad = True
            elif "mlp" in name and mlp == 1:
                param.requires_grad = True
            else:
                param.requires_grad = False

        if use_gpu:
            device = torch.device("cuda:{}".format(0))
            self.gpt2.to(device=device)

        # self.in_layer = nn.Linear(patch_size, d_model)

        self.predict_linear_pre = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
        self.predict_linear = nn.Linear(self.patch_size, enc_in)
        self.ln = nn.LayerNorm(d_ff)
        self.out_layer = nn.Linear(d_ff, c_out)

    def forward(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len :, :]  # [B, L, D]

    def forecast(self, x_enc, x_mark_enc):
        B, L, M = x_enc.shape

        # Normalization from Non-stationary Transformer
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev

        # embedding
        enc_out = self.enc_embedding(x_enc, x_mark_enc)  # [B,T,C]
        enc_out = self.predict_linear_pre(enc_out.permute(0, 2, 1)).permute(
            0, 2, 1
        )  # align temporal dimension
        enc_out = torch.nn.functional.pad(enc_out, (0, 768 - enc_out.shape[-1]))

        # enc_out = rearrange(enc_out, 'b l m -> b m l')
        # enc_out = self.padding_patch_layer(enc_out)
        # enc_out = enc_out.unfold(dimension=-1, size=self.patch_size, step=self.stride)
        # enc_out = self.predict_linear(enc_out)
        # enc_out = rearrange(enc_out, 'b m n p -> b n (m p)')

        dec_out = self.gpt2(inputs_embeds=enc_out).last_hidden_state
        dec_out = dec_out[:, :, : self.d_ff]
        # dec_out = dec_out.reshape(B, -1)

        # dec_out = self.ln(dec_out)
        dec_out = self.out_layer(dec_out)
        # print(dec_out.shape)
        # dec_out = dec_out.reshape(B, self.pred_len + self.seq_len, -1)

        # De-Normalization from Non-stationary Transformer
        dec_out = dec_out * (
            stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )
        dec_out = dec_out + (
            means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len + self.seq_len, 1)
        )

        return dec_out


class GPT4TS(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 0.002,
        loss: str = "SMAPE",
        lradj: str = "type1",
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.criterion = get_loss_fn(loss)
        self.save_hyperparameters(ignore=["model"])

    def model_forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        time = torch.zeros((B, T, 1)).to(device=x.device)
        return self.model(x, time)

    def _shared_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)

        preds = preds[:, :, : prediction_window.shape[-1]]

        assert preds.shape == prediction_window.shape
        mae_criterion = torch.nn.L1Loss()
        mae_loss = mae_criterion(preds, prediction_window)

        loss = self.criterion(
            preds,
            prediction_window,
        )
        return loss, mae_loss

    def model_specific_train_step(self, look_back_window, prediction_window):
        loss, _ = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        loss, mae_loss = self._shared_step(look_back_window, prediction_window)
        if self.tune:
            loss = mae_loss
        self.log("val_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_start(self):
        # Adjust learning rate at the start of each epoch
        adjust_learning_rate(
            self.trainer.optimizers[0],
            self.current_epoch + 1,
            self.hparams.lradj,
            self.hparams.learning_rate,
        )
        current_lr = self.trainer.optimizers[0].param_groups[0]["lr"]
        self.log("current_lr", current_lr, on_epoch=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        return optimizer
