import pandas as pd
import numpy as np
import torch

from IOHMM import UnSupervisedIOHMM
from IOHMM import OLS, DiscreteMNL, CrossEntropyMNL
from torch import Tensor
from typing import Any, Tuple

from src.models.utils import BaseLightningModule


# Dummy model, all train and test logic is in HMMLightningModule
class HMM(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, look_back_window: Tensor) -> Tensor:
        return Tensor([0])


class HMMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        n_states: int = 3,
        n_iter: int = 100,
        em_tol: float = 1e-6,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = UnSupervisedIOHMM(
            num_states=n_states, max_EM_iter=n_iter, EM_tol=em_tol
        )
        self.model.set_models(
            model_emissions=[OLS()],
            model_transition=CrossEntropyMNL(solver="lbfgs"),
            model_initial=CrossEntropyMNL(solver="lbfgs"),
        )

        self.automatic_optimization = False

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset.data

        dfs = []
        for s in train_dataset:
            df = pd.DataFrame(s, columns=["emission"])
            dfs.append(df)

        self.model.set_inputs(
            covariates_initial=[], covariates_transition=[], covariates_emissions=[[]]
        )
        self.model.set_outputs([["emission"]])
        self.model.set_data(dfs)

        self.model.train()

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        batch_size, _, _ = look_back_window.shape
        device = look_back_window.device
        look_back_window = look_back_window.detach().cpu().numpy()  # (B, T, 1)
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
        return Tensor(preds, device=device)

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
