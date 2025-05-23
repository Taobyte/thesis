import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from typing import Tuple

from src.models.utils import BaseLightningModule


class PytorchHMM(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 5,
        target_channel_dim: int = 1,
        temperature: float = 1.0,
        min_var: float = 1e-6,
        use_dynamic_features: bool = False,
        dynamic_exogenous_variables: int = 2,
        ff_dim: int = 32,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.target_channel_dim = target_channel_dim
        self.temperature = temperature
        self.min_var = min_var

        # Transition matrix parameters (learnable)
        self.transition_logits = nn.Parameter(torch.randn(hidden_dim, hidden_dim))

        # Initial state distribution parameters
        self.initial_logits = nn.Parameter(torch.randn(hidden_dim))

        self.use_dynamic_features = use_dynamic_features
        if use_dynamic_features:
            self.dynamic_exogenous_variables = dynamic_exogenous_variables
            self.emission_nns = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.dynamic_exogenous_variables, ff_dim),
                        nn.ReLU(),
                        nn.Linear(
                            ff_dim, 2 * self.target_channel_dim
                        ),  # outputs mean and log-variance
                    )
                    for _ in range(self.hidden_dim)
                ]
            )
        else:
            self.emission_means = nn.Parameter(
                torch.randn(hidden_dim, target_channel_dim)
            )
            self.emission_log_vars = nn.Parameter(
                torch.zeros(hidden_dim, target_channel_dim)
            )

        # Initialize parameters
        self._initialize_parameters()

    def _initialize_parameters(self):
        """Initialize parameters with reasonable defaults."""
        with torch.no_grad():
            # Initialize transition matrix to slightly favor staying in same state
            self.transition_logits.fill_(0.0)
            self.transition_logits.diagonal().fill_(1.0)

            # Initialize initial state distribution uniformly
            self.initial_logits.fill_(0.0)

            # Initialize emission means with different values for each state
            for i in range(self.hidden_dim):
                self.emission_means[i].fill_(i - self.hidden_dim // 2)

            # Initialize emission variances to reasonable values
            self.emission_log_vars.fill_(np.log(1.0))

    @property
    def transition_matrix(self):
        """Get normalized transition matrix."""
        return F.softmax(self.transition_logits / self.temperature, dim=-1)

    @property
    def initial_distribution(self):
        """Get normalized initial state distribution."""
        return F.softmax(self.initial_logits / self.temperature, dim=-1)

    @property
    def emission_variances(self):
        """Get emission variances with minimum threshold."""
        return torch.exp(self.emission_log_vars) + self.min_var

    def compute_emission_probability(
        self, look_back_window: torch.Tensor, state: int
    ) -> torch.Tensor:
        if self.use_dynamic_features:
            activity = look_back_window[
                :,
                self.target_channel_dim : self.target_channel_dim
                + self.dynamic_exogenous_variables,
            ]
            params = self.emission_nns[state](activity)
            mean, log_var = params.chunk(2, dim=-1)
            var = torch.exp(log_var) + self.min_var
        else:
            mean = (
                self.emission_means[state].unsqueeze(0).unsqueeze(0)
            )  # (1, 1, target_channel_dim)
            var = (
                self.emission_variances[state].unsqueeze(0).unsqueeze(0)
            )  # (1, 1, target_channel_dim)

        # Gaussian log probability
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * var) + ((look_back_window - mean) ** 2) / var
        )

        # Sum over output dimensions
        return log_prob.sum(dim=-1)

    def forward_algorithm(
        self, observations: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = observations.shape
        alpha = torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=observations.device
        )

        # Initialize first timestep
        initial_dist = self.initial_distribution.unsqueeze(0).expand(batch_size, -1)
        for s in range(self.hidden_dim):
            emission_prob = self.compute_emission_probability(
                observations[:, 0:1], s
            ).squeeze(1)
            alpha[:, 0, s] = torch.log(initial_dist[:, s] + 1e-10) + emission_prob

        # Forward pass
        transition_matrix = self.transition_matrix
        for t in range(1, seq_len):
            for s in range(self.hidden_dim):
                emission_prob = self.compute_emission_probability(
                    observations[:, t : t + 1], s
                ).squeeze(1)

                # Log-sum-exp for numerical stability
                transition_scores = alpha[:, t - 1, :] + torch.log(
                    transition_matrix[:, s] + 1e-10
                )
                alpha[:, t, s] = (
                    torch.logsumexp(transition_scores, dim=-1) + emission_prob
                )

        # Compute total log likelihood
        log_likelihood = torch.logsumexp(alpha[:, -1, :], dim=-1)

        return alpha, log_likelihood

    def viterbi_decode(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = observations.shape
        delta = torch.zeros(
            batch_size, seq_len, self.hidden_dim, device=observations.device
        )
        psi = torch.zeros(
            batch_size,
            seq_len,
            self.hidden_dim,
            dtype=torch.long,
            device=observations.device,
        )

        # Initialize
        initial_dist = self.initial_distribution.unsqueeze(0).expand(batch_size, -1)
        for s in range(self.hidden_dim):
            emission_prob = self.compute_emission_probability(
                observations[:, :1], s
            ).squeeze(1)
            delta[:, 0, s] = torch.log(initial_dist[:, s] + 1e-10) + emission_prob

        # Forward pass
        transition_matrix = self.transition_matrix
        for t in range(1, seq_len):
            for s in range(self.hidden_dim):
                emission_prob = self.compute_emission_probability(
                    observations[:, t : t + 1], s
                ).squeeze(1)

                transitions = delta[:, t - 1, :] + torch.log(
                    transition_matrix[:, s] + 1e-10
                )
                psi[:, t, s] = torch.argmax(transitions, dim=-1)
                delta[:, t, s] = torch.max(transitions, dim=-1)[0] + emission_prob

        # Backtrack
        states = torch.zeros(
            batch_size, seq_len, dtype=torch.long, device=observations.device
        )
        states[:, -1] = torch.argmax(delta[:, -1, :], dim=-1)

        for t in range(seq_len - 2, -1, -1):
            states[:, t] = psi[range(batch_size), t + 1, states[:, t + 1]]

        return states

    def infer_current_state(self, observations: torch.Tensor) -> torch.Tensor:
        alpha, _ = self.forward_algorithm(observations)
        # Normalize the final forward probabilities
        final_alpha = alpha[:, -1, :]  # (batch_size, hidden_dim)
        state_probs = F.softmax(final_alpha, dim=-1)
        return state_probs

    def predict_next_timestep(
        self, current_state_probs: torch.Tensor, deterministic: bool = False
    ) -> torch.Tensor:
        # Compute next state probabilities
        transition_matrix = self.transition_matrix
        next_state_probs = torch.matmul(current_state_probs, transition_matrix)

        if deterministic:
            # Expected value prediction
            predictions = torch.sum(
                next_state_probs.unsqueeze(-1) * self.emission_means.unsqueeze(0), dim=1
            ).unsqueeze(1)  # (batch_size, 1, target_channel_dim)
        else:
            # Sample from mixture of Gaussians
            # Sample states
            state_samples = torch.multinomial(next_state_probs, 1).squeeze(
                -1
            )  # (batch_size,)

            # Sample from emission distributions
            selected_means = self.emission_means[
                state_samples
            ]  # (batch_size, target_channel_dim)
            selected_vars = self.emission_variances[
                state_samples
            ]  # (batch_size, target_channel_dim)

            noise = torch.randn_like(selected_means)
            predictions = (
                selected_means + torch.sqrt(selected_vars) * noise
            ).unsqueeze(1)

        return predictions, next_state_probs

    def autoregressive_predict(
        self,
        lookback_sequence: torch.Tensor,
        prediction_steps: int,
        deterministic: bool = True,
    ) -> torch.Tensor:
        predictions = []

        # Infer current state from lookback sequence
        current_state_probs = self.infer_current_state(lookback_sequence)

        for _ in range(prediction_steps):
            # Predict next timestep
            pred, current_state_probs = self.predict_next_timestep(
                current_state_probs, deterministic
            )
            predictions.append(pred)

        return torch.cat(
            predictions, dim=1
        )  # (batch_size, prediction_steps, target_channel_dim)


class HMMLightningModule(BaseLightningModule):
    """
    PyTorch Lightning wrapper for the HMM model.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        deterministic: bool = True,
    ):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.deterministic = deterministic

        self.model = model

        self.criterion = torch.nn.MSELoss()

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        prediction_window = getattr(self.trainer.datamodule, "prediction_window")

        return self.model.autoregressive_predict(
            look_back_window, prediction_window, deterministic=True
        )

    def model_specific_train_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
        target_channel_dim = getattr(self.trainer.datamodule, "target_channel_dim")
        full_sequence = torch.cat(
            [look_back_window[:, :, :target_channel_dim], prediction_window], dim=1
        )

        _, log_likelihood = self.model.forward_algorithm(full_sequence)

        loss = -log_likelihood.mean()

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def model_specific_val_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
        # _, log_likelihood = self.model.forward_algorithm(look_back_window)

        preds = self.model_forward(look_back_window)
        mse_loss = self.criterion(preds, prediction_window)

        self.log("val_loss", mse_loss, on_step=False, on_epoch=True, prog_bar=True)

        return mse_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
