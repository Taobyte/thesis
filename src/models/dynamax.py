# This model builds on the Dynamax library for probabilistic state space models:
# https://github.com/probml/dynamax

import torch
import jax
import jax.numpy as jnp
import jax.random as jr
import optax

from jax import lax
from jax.tree_util import tree_map
from jaxtyping import Int, Float, Array
from jax import vmap
from dynamax.hidden_markov_model import GaussianHMM, LinearAutoregressiveHMM
from dynamax.linear_gaussian_ssm.models import LinearGaussianSSM
from dynamax.hidden_markov_model.models.abstractions import (
    HMMParameterSet,
)
from torch import Tensor
from typing import Any, Tuple, Optional

from src.models.utils import BaseLightningModule


# Dummy model, all train and test logic is in HMMLightningModule
class Dummy(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, look_back_window: Tensor) -> Tensor:
        return Tensor([0])


class LARModel(LinearAutoregressiveHMM):
    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    def viterbi_sample(
        self,
        params: HMMParameterSet,
        key: Array,
        num_timesteps: int,
        prev_emissions: Optional[Float[Array, "num_lags emission_dim"]] = None,
    ) -> Tuple[
        Int[Array, " num_timesteps"], Float[Array, "num_timesteps emission_dim"]
    ]:
        r"""Sample states $z_{1:T}$ and emissions $y_{1:T}$ given parameters $\theta$.

        Args:
            params: model parameters $\theta$
            key: random number generator
            num_timesteps: number of timesteps $T$
            prev_emissions: (optionally) preceding emissions $y_{-L+1:0}$. Defaults to zeros.

        Returns:
            latent states and emissions

        """
        assert prev_emissions is not None

        def _step(carry, key):
            """Sample the next state and emission."""
            prev_state, prev_emissions = carry
            key1, key2 = jr.split(key, 2)
            state = self.transition_distribution(
                params, prev_state
            ).mode()  # take argmax states instead of sampling
            emission = self.emission_distribution(
                params, state, inputs=jnp.ravel(prev_emissions)
            ).mean()  # take mean instead of sampling
            next_prev_emissions = jnp.vstack([emission, prev_emissions[:-1]])
            return (state, next_prev_emissions), (state, emission)

        # Sample the initial state
        lagged_emissions = self.compute_inputs(emissions=prev_emissions)

        initial_state = self.most_likely_states(
            params, emissions=prev_emissions, inputs=lagged_emissions
        )[-1]

        initial_emission = self.emission_distribution(
            params, initial_state, inputs=jnp.ravel(prev_emissions)
        ).mean()
        initial_prev_emissions = jnp.vstack([initial_emission, prev_emissions[:-1]])

        # Sample the remaining emissions and states
        key1, key2, key = jr.split(key, 3)
        next_keys = jr.split(key, num_timesteps - 1)
        _, (next_states, next_emissions) = lax.scan(
            _step, (initial_state, initial_prev_emissions), next_keys
        )

        # Concatenate the initial state and emission with the following ones
        expand_and_cat = lambda x0, x1T: jnp.concatenate((jnp.expand_dims(x0, 0), x1T))
        states = tree_map(expand_and_cat, initial_state, next_states)
        emissions = tree_map(expand_and_cat, initial_emission, next_emissions)
        return states, emissions


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
            self.model = LARModel(
                num_states=n_states,
                emission_dim=look_back_channel_dim,
                num_lags=look_back_window,
                transition_matrix_stickiness=transition_matrix_stickiness,
            )
        elif model_type == "LinearGaussianSSM":
            self.model = LinearGaussianSSM(
                n_states,
                emission_dim=target_channel_dim,
                input_dim=look_back_channel_dim - target_channel_dim,
            )
        else:
            raise NotImplementedError()

        self.automatic_optimization = False
        self.val_criterion = torch.nn.L1Loss()

    def on_train_epoch_start(self):
        datamodule = self.trainer.datamodule
        assert datamodule.name in ["fieee", "fdalia", "fwildppg"]

        ts = datamodule.train_dataset.get_normalized_timeseries()
        emissions = jnp.array(ts)

        key = jr.PRNGKey(self.seed)
        if self.model_type == "GaussianHMM":
            init_params, props = self.model.initialize(key)

            em_params, _ = self.model.fit_em(
                init_params, props, emissions, num_iters=self.n_iter
            )
        elif self.model_type == "LinearAutoregressiveHMM":
            assert isinstance(self.model, LinearAutoregressiveHMM)

            inputs = self.model.compute_inputs(emissions)
            emissions = emissions[self.look_back_window :, :]
            inputs = inputs[self.look_back_window :, :]

            params, props = self.model.initialize(
                key=key, method="kmeans", emissions=emissions
            )

            if self.optimizer == "em":
                em_params, loglikelihood = self.model.fit_em(
                    params, props, emissions, inputs=inputs, num_iters=self.n_iter
                )
                print(f"Log-Likelihood: {loglikelihood}")

            elif self.optimizer == "sgd":
                assert self.n_batch <= len(train_dataset)
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
                _, emissions = self.model.viterbi_sample(
                    self.em_params,
                    key=subkey,
                    num_timesteps=self.prediction_window,
                    prev_emissions=prev_em,
                )
                return emissions

            self.batched_sample_fn = vmap(sample_fn, in_axes=(0, 0))

        elif self.model_type == "LinearGaussianSSM":
            assert isinstance(self.model, LinearGaussianSSM)

            endo = emissions[:, : self.target_channel_dim]  # endo
            inputs = emissions[:, self.target_channel_dim :]  # exo

            params, props = self.model.initialize(key=key)
            em_params, loglikelihood = self.model.fit_em(
                params, props, endo, inputs=inputs, num_iters=self.n_iter
            )
            print(f"Log-Likelihood: {loglikelihood}")
            self.em_params = em_params

            def forecast_fn(endo, exo):
                _, _, preds, _ = self.model.forecast(
                    params,
                    emissions=endo,
                    num_forecast_timesteps=self.prediction_window,
                    inputs=exo,
                )
                return preds

            self.batched_forecast = vmap(forecast_fn, in_axes=(0, 0))

        self.em_params = em_params
        self.key = jr.PRNGKey(self.seed)

    def model_forward(self, look_back_window: torch.Tensor) -> Tensor:
        batch_size, _, _ = look_back_window.shape
        device = look_back_window.device
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
        elif self.model_type == "LinearGaussianSSM":
            assert isinstance(self.model, LinearGaussianSSM)
            endo = look_back_window_jax[:, :, : self.target_channel_dim]
            exo = look_back_window_jax[:, :, self.target_channel_dim :]
            preds = self.batched_forecast(endo, exo)

        return torch.tensor(jax.device_get(preds), device=device)

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
        if self.model_type == "LinearAutoregressiveHMM":
            preds = self.batched_sample_fn(keys, look_back_window_jax)
            preds = torch.tensor(jax.device_get(preds), device=device)
            preds = preds[:, :, :C]
        elif self.model_type == "LinearGaussianSSM":
            endo = look_back_window_jax[:, :, : self.target_channel_dim]
            exo = look_back_window_jax[:, :, self.target_channel_dim :]
            preds = self.batched_forecast(endo, exo)
            preds = torch.tensor(jax.device_get(preds), device=device)
            preds = preds[:, :, :C]

        val_loss = self.val_criterion(preds, prediction_window_norm)
        self.log("val_loss", val_loss, on_epoch=True, on_step=True, logger=True)

    def configure_optimizers(self):
        return None

    def cleanup_jax_memory(self):
        """Clean up JAX memory and reset state"""
        # Clear JAX compilation cache
        jax.clear_caches()

        # Force garbage collection of JAX arrays
        import gc

        gc.collect()

        # Reset PRNG key
        self.key = jr.PRNGKey(self.seed)

        # Clear any cached functions
        if hasattr(self, "batched_sample_fn"):
            delattr(self, "batched_sample_fn")
        if hasattr(self, "em_params"):
            delattr(self, "em_params")
