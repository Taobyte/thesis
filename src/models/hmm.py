import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from typing import Tuple

from src.models.utils import BaseLightningModule


class ExoToMatrix(nn.Module):
    def __init__(self, exo_dim: int, base_dim: int):
        super().__init__()
        self.base_dim = base_dim
        self.linear = nn.Linear(exo_dim, base_dim**2)

    def forward(self, x):
        B, T, C = x.shape
        x_reshaped = rearrange(x, "B T C -> (B T) C")
        output = self.linear(x_reshaped)
        output_reshaped = rearrange(
            output, "(B T) (E F) -> B T E F", B=B, T=T, E=self.base_dim, F=self.base_dim
        )
        return output_reshaped


class PytorchHMM(nn.Module):
    def __init__(
        self,
        base_dim: int = 5,
        target_channel_dim: int = 1,
        look_back_channel_dim: int = 2,
        look_back_window: int = 5,
        prediction_window: int = 3,
        temperature: float = 1.0,
        min_var: float = 1e-6,
        use_exo: bool = False,
    ):
        super().__init__()

        self.hidden_dim = base_dim
        self.use_exo = use_exo
        self.target_channel_dim = target_channel_dim
        self.look_back_channel_dim = look_back_channel_dim
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.exo_dim = look_back_channel_dim - target_channel_dim
        self.temperature = temperature
        self.min_var = min_var

        # Transition matrix parameters (learnable)
        self.base_transition_logits = nn.Parameter(
            torch.randn(self.hidden_dim, self.hidden_dim)
        )
        if use_exo:
            self.exo_linear = ExoToMatrix(self.exo_dim, base_dim)
            self.linear_exo_forecast = nn.Linear(
                look_back_channel_dim * look_back_window,
                prediction_window * self.exo_dim,
            )

            self.linear_exo_mean = nn.Linear(self.exo_dim, self.target_channel_dim)
            self.linear_exo_std = nn.Linear(self.exo_dim, self.target_channel_dim**2)

        # Initial state distribution parameters
        self.initial_logits = nn.Parameter(torch.randn(self.hidden_dim))

        self.emission_means = nn.Parameter(
            torch.randn(self.hidden_dim, target_channel_dim)
        )
        self.emission_log_vars = nn.Parameter(
            torch.zeros(self.hidden_dim, target_channel_dim)
        )

    def get_transition_matrices(
        self, exo_logits: torch.Tensor = None, batch_size: int = 0
    ):
        """Get normalized transition matrices for all exogenous states or base."""
        if self.use_exo and exo_logits is not None:
            base_logits = self.base_transition_logits  # (B, hidden_dim, hidden_dim)
            # Combine base and exogenous
            combined_logits = base_logits + exo_logits  # (B, hidden_dim, hidden_dim)
            return F.softmax(combined_logits / self.temperature, dim=-1)
        else:
            # Just base transition matrix
            base_matrix = F.softmax(
                self.base_transition_logits / self.temperature, dim=-1
            )
            return base_matrix.unsqueeze(0).expand(batch_size, -1, -1)

    @property
    def initial_distribution(self):
        """Get normalized initial state distribution."""
        return F.softmax(self.initial_logits / self.temperature, dim=-1)

    def get_emission_means(self, exo_variables=None):
        base_mean = self.emission_means.unsqueeze(0).unsqueeze(0)
        if self.use_exo:
            exo_mean_delta = self.linear_exo_mean(
                exo_variables
            )  # (B, T, target_channel_dim)
            B, T, target_channel_dim = exo_mean_delta.shape
            exo_mean_delta = exo_mean_delta.unsqueeze(2).expand(
                B, T, self.hidden_dim, target_channel_dim
            )

            mean = base_mean + exo_mean_delta
        else:
            mean = base_mean
        return mean  # (B, T, hidden_dim, target_channel_dim)

    def get_emission_variances(self, exo_variables=None):
        base_var = torch.exp(self.emission_log_vars).unsqueeze(0).unsqueeze(0)
        if self.use_exo:
            log_var = self.linear_exo_std(exo_variables)  # (B, T, target_channel_dim)
            B, T, target_channel_dim = log_var.shape
            log_var = log_var.unsqueeze(2).expand(
                B, T, self.hidden_dim, target_channel_dim
            )
            exo_var_delta = torch.exp(log_var)
            var = torch.exp(self.emission_log_vars) + exo_var_delta
        else:
            var = base_var

        return var + self.min_var

    def compute_emission_probabilities(
        self, observations: torch.Tensor
    ) -> torch.Tensor:
        """Compute emission log probabilities for all states and time steps."""
        batch_size, seq_len, obs_dim = observations.shape

        if self.use_exo:
            exo_variables = observations[
                :, :, self.target_channel_dim :
            ]  # (B, T, C_exo)
        else:
            exo_variables = None

        # observations: [batch, seq_len, obs_dim] -> [batch, seq_len, 1, obs_dim]
        obs_expanded = observations.unsqueeze(2)

        # means: [hidden_dim, target_channel_dim] -> [1, 1, hidden_dim, target_channel_dim]

        means_expanded = self.get_emission_means(exo_variables)

        # vars: [hidden_dim, target_channel_dim] -> [1, 1, hidden_dim, target_channel_dim]
        vars_expanded = self.get_emission_variances(exo_variables)

        # Compute Gaussian log probability for all states at once
        # [batch, seq_len, hidden_dim, target_channel_dim]
        log_prob = -0.5 * (
            torch.log(2 * torch.pi * vars_expanded)
            + ((obs_expanded - means_expanded) ** 2) / vars_expanded
        )

        # Sum over target channel dimensions
        # [batch, seq_len, hidden_dim]
        return log_prob.sum(dim=-1)

    def forward_algorithm(
        self,
        observations: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, seq_len, _ = observations.shape
        device = observations.device

        if self.use_exo:
            exo_variables = observations[:, :, self.target_channel_dim :]
            exo_logits = self.exo_linear(exo_variables)  # (B, hidden_dim, hidden_dim)

        # Compute all emission probabilities at once
        # [batch, seq_len, hidden_dim]
        emission_log_probs = self.compute_emission_probabilities(observations)

        # Initialize alpha (log probabilities)
        alpha = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)

        # Initialize first timestep
        initial_dist = self.initial_distribution  # [hidden_dim]
        alpha[:, 0, :] = torch.log(initial_dist + 1e-10) + emission_log_probs[:, 0, :]

        batch_size = observations.shape[0]
        for t in range(1, seq_len):
            if self.use_exo:
                transition_matrix = self.get_transition_matrices(exo_logits[:, t - 1])
            else:
                transition_matrix = self.get_transition_matrices(batch_size=batch_size)
            trans_log = torch.log(
                transition_matrix + 1e-10
            )  # [batch, hidden_dim, hidden_dim]

            # Expand alpha for broadcasting: [batch, hidden_dim, 1]
            alpha_prev = alpha[:, t - 1, :].unsqueeze(2)

            # Compute transition scores: [batch, hidden_dim, hidden_dim]
            transition_scores = alpha_prev + trans_log

            # Log-sum-exp over previous states: [batch, hidden_dim]
            alpha[:, t, :] = (
                torch.logsumexp(transition_scores, dim=1) + emission_log_probs[:, t, :]
            )

        # Compute total log likelihood
        log_likelihood = torch.logsumexp(alpha[:, -1, :], dim=-1)
        return alpha, log_likelihood

    def infer_current_state(self, observations: torch.Tensor) -> torch.Tensor:
        alpha, log_likelihood = self.forward_algorithm(observations)
        # Normalize the final forward probabilities
        final_alpha = alpha[:, -1, :]  # (batch_size, self.hidden_dim)
        state_probs = F.softmax(final_alpha, dim=-1)
        return state_probs, log_likelihood

    def predict_next_timestep(
        self,
        current_state_probs: torch.Tensor,
        deterministic: bool = False,
        last_ex_variable: torch.Tensor = None,
    ) -> torch.Tensor:
        batch_size, _ = current_state_probs.shape
        if self.use_exo:
            exo_logits = self.exo_linear(last_ex_variable)
        else:
            exo_logits = None
        transition_matrix = self.get_transition_matrices(
            exo_logits=exo_logits, batch_size=batch_size
        )

        if self.use_exo:
            transition_matrix = transition_matrix.squeeze(1)

        next_state_probs = torch.bmm(
            current_state_probs.unsqueeze(1), transition_matrix
        ).squeeze(1)

        if deterministic:
            # take most probably state and use the mean
            state_samples = torch.argmax(next_state_probs, dim=-1)  # (B, )
            predictions = self.emission_means[state_samples].unsqueeze(
                1
            )  # (B, 1, target_channel_dim)
            if self.use_exo:
                # add delta mean from the exogenous variables
                reshaped_ex = rearrange(
                    last_ex_variable, "B T C -> B (T C)"
                )  # in here T = 1
                delta_mean = self.linear_exo_mean(
                    reshaped_ex
                )  # (B, 1 * target_channel_dim)
                predictions = predictions + delta_mean.unsqueeze(1)

        else:
            state_samples = torch.multinomial(next_state_probs, 1).squeeze(
                -1
            )  # (batch_size,)

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

        current_state_probs, log_likelihood = self.infer_current_state(
            lookback_sequence
        )

        if self.use_exo:
            last_exo_lbw = lookback_sequence[
                :, -1, self.target_channel_dim :
            ].unsqueeze(1)  # (B, 1, C_exo)
            input = rearrange(lookback_sequence, "B T C -> B (T C)")
            output = self.linear_exo_forecast(input)  # (B, L * C_exo)
            predicted_exo = rearrange(
                output, "B (L C) -> B L C", L=self.prediction_window
            )

            exo = torch.concat((last_exo_lbw, predicted_exo), dim=1)

        for i in range(prediction_steps):
            # Predict next timestep
            pred, current_state_probs = self.predict_next_timestep(
                current_state_probs,
                deterministic,
                exo[:, i, :].unsqueeze(1) if self.use_exo else None,
            )
            predictions.append(pred)

        return torch.cat(
            predictions, dim=1
        ), log_likelihood  # (batch_size, prediction_steps, target_channel_dim)


class HMMLightningModule(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        learning_rate: float = 1e-3,
        deterministic: bool = True,
        pred_loss_weight: float = 1.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.deterministic = deterministic
        self.pred_loss_weight = pred_loss_weight

        self.model = model

        self.criterion = torch.nn.MSELoss()

    def model_forward(self, look_back_window: torch.Tensor) -> torch.Tensor:
        prediction_window_length = getattr(self.trainer.datamodule, "prediction_window")
        preds, _ = self.model.autoregressive_predict(
            look_back_window,
            prediction_window_length,
            deterministic=self.deterministic,
        )
        return preds

    def model_specific_train_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
        prediction_window_length = prediction_window.shape[1]
        # full_sequence = torch.cat([look_back_window[:, :, :], prediction_window], dim=1)
        # TODO: currently we do not train with all data!

        # _, log_likelihood = self.model.forward_algorithm(look_back_window)
        # loss = -log_likelihood.mean()

        preds, log_likelihood = self.model.autoregressive_predict(
            look_back_window, prediction_window_length, deterministic=self.deterministic
        )
        ll_loss = -log_likelihood.mean()
        mse_loss = self.criterion(preds, prediction_window)
        loss = mse_loss * self.pred_loss_weight + ll_loss * (1 - self.pred_loss_weight)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def model_specific_val_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ) -> torch.Tensor:
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


"""

    def viterbi_decode(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = observations.shape
        device = observations.device

        # Extract exogenous variables if using activity labels
        if self.use_activity_labels:
            exog_vars = torch.argmax(
                observations[:, :-1, self.target_channel_dim :], dim=-1
            )
        else:
            exog_vars = None

        # Compute emission probabilities
        emission_log_probs = self.compute_emission_probabilities(
            observations[:, :, : self.target_channel_dim]
        )

        # Get transition matrices
        if self.use_activity_labels:
            transition_matrices = self.get_transition_matrices(exog_vars)
        else:
            transition_matrices = self.get_transition_matrices()

        # Initialize delta and path tracking
        delta = torch.zeros(batch_size, seq_len, self.hidden_dim, device=device)
        path_indices = torch.zeros(
            batch_size, seq_len - 1, self.hidden_dim, dtype=torch.long, device=device
        )

        # Initialize first timestep
        initial_dist = self.initial_distribution
        delta[:, 0, :] = torch.log(initial_dist + 1e-10) + emission_log_probs[:, 0, :]

        # Forward pass
        for t in range(1, seq_len):
            if self.use_activity_labels:
                trans_t = transition_matrices[:, t - 1, :, :]
                trans_log_t = torch.log(trans_t + 1e-10)

                # [batch, hidden_dim, 1] + [batch, hidden_dim, hidden_dim] -> [batch, hidden_dim, hidden_dim]
                scores = delta[:, t - 1, :].unsqueeze(2) + trans_log_t

                # Find best previous state for each current state
                delta[:, t, :], path_indices[:, t - 1, :] = torch.max(scores, dim=1)
                delta[:, t, :] += emission_log_probs[:, t, :]
            else:
                trans_log = torch.log(transition_matrices + 1e-10)

                # [batch, hidden_dim, 1] + [1, hidden_dim, hidden_dim] -> [batch, hidden_dim, hidden_dim]
                scores = delta[:, t - 1, :].unsqueeze(2) + trans_log.unsqueeze(0)

                delta[:, t, :], path_indices[:, t - 1, :] = torch.max(scores, dim=1)
                delta[:, t, :] += emission_log_probs[:, t, :]

        # Backtrack to find best path
        best_paths = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)

        # Find best final state
        _, best_paths[:, -1] = torch.max(delta[:, -1, :], dim=1)

        # Backtrack
        for t in range(seq_len - 2, -1, -1):
            best_paths[:, t] = path_indices[
                torch.arange(batch_size), t, best_paths[:, t + 1]
            ]

        return best_paths


"""
