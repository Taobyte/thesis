import torch
import jax.numpy as jnp
import jax.random as jr
import optax

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
        look_back_channel_dim: int = 2,
        look_back_window: int = 5,
        n_states: int = 3,
        n_iter: int = 100,
        model_type: str = "GaussianHMM",
        transition_matrix_stickiness: float = 10.0,
        optimizer: str = "em",
        learning_rate: float = 0.001,
        momentum: float = 0.95,
        n_batch: int = 2,
        deterministic: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.n_states = n_states
        self.n_iter = n_iter
        self.model_type = model_type
        self.deterministic = deterministic
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.n_batch = n_batch

        if model_type == "GaussianHMM":
            assert self.experiment_name == "endo_only"
            self.model = GaussianHMM(
                n_states,
                emission_dim=target_channel_dim,
                transition_matrix_stickiness=transition_matrix_stickiness,
            )
        elif model_type == "LinearAutoregressiveHMM":
            self.model = LinearAutoregressiveHMM(
                n_states,
                emission_dim=look_back_channel_dim,
                num_lags=look_back_window,
                transition_matrix_stickiness=transition_matrix_stickiness,
            )
        else:
            raise NotImplementedError()

        self.automatic_optimization = False
        self.val_criterion = torch.nn.L1Loss()

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        train_dataset = datamodule.get_numpy_normalized("train")
        min_length = min([len(s) for s in train_dataset])
        emissions = jnp.stack([s[:min_length, :] for s in train_dataset], axis=0)
        # emissions = [jnp.array(s) for s in train_dataset]

        assert self.n_batch <= len(train_dataset)

        key = jr.PRNGKey(self.seed)
        if self.model_type == "GaussianHMM":
            init_params, props = self.model.initialize(key)

            em_params, _ = self.model.fit_em(
                init_params, props, emissions, num_iters=self.n_iter
            )
        elif self.model_type == "LinearAutoregressiveHMM":
            assert isinstance(self.model, LinearAutoregressiveHMM)
            inputs = vmap(self.model.compute_inputs, in_axes=0)(emissions)
            emissions = emissions[:, self.look_back_window :, :]
            inputs = inputs[:, self.look_back_window :, :]

            params, props = self.model.initialize(
                key=key, method="kmeans", emissions=emissions
            )

            if self.optimizer == "em":
                em_params, loglikelihood = self.model.fit_em(
                    params, props, emissions, inputs=inputs, num_iters=self.n_iter
                )
                print(f"Log-Likelihood: {loglikelihood}")

            elif self.optimizer == "sgd":
                sgd_key = jr.PRNGKey(0)
                em_params, sgd_losses = self.model.fit_sgd(
                    params,
                    props,
                    emissions,
                    optimizer=optax.sgd(
                        learning_rate=self.learning_rate, momentum=self.momentum
                    ),
                    # batch_size=self.n_batch,
                    num_epochs=self.n_iter,
                    key=sgd_key,
                    inputs=inputs,
                )
                print(f"SGD losses: {sgd_losses}")

            def sample_fn(subkey, prev_em):
                _, emissions = self.model.sample(
                    params,
                    key=subkey,
                    num_timesteps=self.prediction_window,
                    prev_emissions=prev_em,
                    deterministic=self.deterministic,
                )
                return emissions

            self.batched_sample_fn = vmap(sample_fn, in_axes=(0, 0))

        self.em_params = em_params
        self.key = jr.PRNGKey(self.seed)

    def model_forward(self, look_back_window: torch.Tensor) -> Tensor:
        batch_size, _, _ = look_back_window.shape
        look_back_window_jax = jnp.array(look_back_window.detach().cpu().numpy())
        params = self.em_params

        if self.model_type == "GaussianHMM":
            preds = []

            for b in range(batch_size):
                seq = look_back_window_jax[b]

                posterior = self.model.smoother(params, seq)

                final_state_probs = posterior.smoothed_probs[-1]  # (n_states,)

                key, subkey = jr.split(self.key)
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

                    emission = jr.multivariate_normal(
                        subkey, emission_mean, emission_cov
                    )
                    batch_preds.append(emission)

                    current_state = next_state

                preds.append(jnp.stack(batch_preds, axis=0))

            preds = jnp.stack(preds, axis=0)
        elif self.model_type == "LinearAutoregressiveHMM":
            assert isinstance(self.model, LinearAutoregressiveHMM)
            keys = jr.split(self.key, batch_size)
            preds = self.batched_sample_fn(keys, look_back_window_jax)

        return torch.tensor(preds)

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        return torch.Tensor()

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor], batch_idx: int
    ):
        _, look_back_window_norm, prediction_window_norm = batch
        batch_size, _, C = prediction_window_norm.shape
        device = prediction_window_norm.device
        look_back_window_jax = jnp.array(look_back_window_norm.detach().cpu().numpy())
        keys = jr.split(self.key, batch_size)
        preds = self.batched_sample_fn(keys, look_back_window_jax)
        preds = Tensor(preds, device=device)
        preds = preds[:, :, :C]
        val_loss = self.val_criterion(preds, prediction_window_norm)
        self.log("val_loss", val_loss, on_epoch=True, on_step=True, logger=True)

    def configure_optimizers(self):
        return None
