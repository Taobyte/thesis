import numpy as np
import torch
import jax.numpy as jnp
import jax.random as jr

from jax import vmap
from dynamax.hidden_markov_model import GaussianHMM, LinearAutoregressiveHMM
from torch import Tensor
from typing import Any, Tuple

from src.models.utils import BaseLightningModule


# Dummy model, all train and test logic is in HMMLightningModule
class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, look_back_window: Tensor) -> Tensor:
        return Tensor([0])


class DynamaxLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        target_channel_dim: int = 1,
        look_back_window: int = 5,
        n_states: int = 3,
        n_iter: int = 100,
        model_type: str = "GaussianHMM",
        transition_matrix_stickiness: float = 10.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        assert self.experiment_name == "endo_only"

        self.n_states = n_states
        self.n_iter = n_iter
        self.model_type = model_type

        if model_type == "GaussianHMM":
            self.model = GaussianHMM(
                n_states,
                emission_dim=target_channel_dim,
                transition_matrix_stickiness=transition_matrix_stickiness,
            )
        elif model_type == "LinearAutoregressiveHMM":
            self.model = LinearAutoregressiveHMM(
                n_states,
                emission_dim=target_channel_dim,
                num_lags=look_back_window,
                transition_matrix_stickiness=transition_matrix_stickiness,
            )
        else:
            raise NotImplementedError()

        self.automatic_optimization = False

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        train_dataset = datamodule.train_dataset.data
        min_length = min([len(s) for s in train_dataset])
        emissions = jnp.stack([s[:min_length, :] for s in train_dataset], axis=0)

        key = jr.PRNGKey(self.seed)
        if self.model_type == "GaussianHMM":
            init_params, props = self.model.initialize(key)

            em_params, log_probs = self.model.fit_em(
                init_params, props, emissions, num_iters=self.n_iter
            )
        elif self.model_type == "LinearAutoregressiveHMM":
            assert isinstance(self.model, LinearAutoregressiveHMM)
            inputs = vmap(self.model.compute_inputs)(emissions)
            params, props = self.model.initialize(
                key=key, method="kmeans", emissions=emissions
            )
            em_params, _ = self.model.fit_em(params, props, emissions, inputs=inputs)

        self.em_params = em_params

    def model_forward(self, look_back_window: torch.Tensor) -> Tensor:
        batch_size, _, _ = look_back_window.shape
        look_back_window_jax = jnp.array(look_back_window.detach().cpu().numpy())
        params = self.em_params
        key = jr.PRNGKey(self.seed)

        if self.model_type == "GaussianHMM":
            preds = []

            for b in range(batch_size):
                seq = look_back_window_jax[b]

                posterior = self.model.smoother(params, seq)

                final_state_probs = posterior.smoothed_probs[-1]  # (n_states,)

                key, subkey = jr.split(key)
                final_state = jr.choice(subkey, self.n_states, p=final_state_probs)

                current_state = final_state
                batch_preds = []

                for h in range(self.prediction_window):
                    key, subkey = jr.split(key)
                    transition_probs = params.transitions.transition_matrix[
                        current_state
                    ]
                    next_state = jr.choice(subkey, self.n_states, p=transition_probs)

                    key, subkey = jr.split(key)
                    emission_mean = params.emissions.means[next_state]
                    emission_cov = params.emissions.covs[next_state]

                    # Sample from multivariate normal
                    emission = jr.multivariate_normal(
                        subkey, emission_mean, emission_cov
                    )
                    batch_preds.append(emission)

                    current_state = next_state

                preds.append(jnp.stack(batch_preds, axis=0))  # (H, emission_dim)

            preds = jnp.stack(preds, axis=0)  # (B, H, emission_dim)
        elif self.model_type == "LinearAutoregressiveHMM":
            assert isinstance(self.model, LinearAutoregressiveHMM)
            preds = []
            for i in range(batch_size):
                sampled_states, sampled_emissions = self.model.sample(
                    params,
                    key,
                    self.prediction_window,
                    prev_emissions=look_back_window_jax[i],
                )

                preds.append(sampled_emissions)

            preds = jnp.stack(preds)
        return torch.tensor(preds)

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
