import numpy as np
import torch

from hmmlearn import hmm
from torch import Tensor
from typing import Any, Tuple

from src.models.utils import BaseLightningModule


class HMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.remodel = hmm.GaussianHMM(
            n_components=3, covariance_type="full", n_iter=100
        )

    def forward(self, look_back_window: Tensor) -> Tensor:
        return Tensor([0])


class HMMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        n_states: int = 3,
        n_iter: int = 100,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert self.experiment_name == "endo_only"

        self.criterion = torch.nn.MSELoss()

        self.remodel = hmm.GaussianHMM(
            n_components=n_states, covariance_type="full", n_iter=n_iter, verbose=True
        )

        self.automatic_optimization = False

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset.data
        X = np.concatenate(train_dataset, axis=0)
        lengths = [len(s) for s in train_dataset]
        self.remodel.fit(X, lengths)

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = look_back_window.shape
        look_back_window = look_back_window.detach().numpy()  # (B, T, 1)
        # TODO: can maybe be done more efficiently
        preds = []
        for i in range(batch_size):
            lbw = look_back_window[i, :, :]
            state_sequence = self.remodel.predict(lbw)
            last_state = state_sequence[-1]
            samples, states = self.remodel.sample(
                n_samples=self.prediction_window,
                random_state=None,
                currstate=last_state,
            )
            preds.append(samples)

        preds = np.stack(preds, axis=0)
        return Tensor(preds)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return torch.Tensor()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        self.log("val_loss", 0, on_epoch=True, on_step=True, logger=True)

    def configure_optimizers(self):
        return None
