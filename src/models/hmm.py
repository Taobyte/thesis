import torch
import torch.nn.functional as F

from einops import rearrange
from torch import Tensor
from typing import Tuple, Any

from src.models.utils import BaseLightningModule


class LinearAutoregressiveHMM(torch.nn.Module):
    def __init__(
        self,
        n_states: int = 2,
        look_back_window: int = 5,
        prediction_window: int = 3,
        look_back_channel_dim: int = 2,
    ):
        super().__init__()
        self.n_states = n_states
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.look_back_channel_dim = look_back_channel_dim

        # HMM parameters
        self.transition_matrix = torch.nn.Parameter(torch.randn(n_states, n_states))
        self.initial_distribution = torch.nn.Parameter(
            torch.randn(
                n_states,
            )
        )

        # Multivariate Gaussian Emission Distribution parameters
        self.means = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(
                        look_back_channel_dim,
                    )
                )
                for _ in range(n_states)
            ]
        )
        # Covariance matrices for each state (using Cholesky decomposition for positive definiteness)
        self.covar_chol = torch.nn.ParameterList(
            [
                torch.nn.Parameter(torch.eye(look_back_channel_dim))
                for _ in range(n_states)
            ]
        )

        # Autoregressive weights
        self.W = torch.nn.ParameterList(
            [
                torch.nn.Parameter(
                    torch.randn(
                        look_back_window, look_back_channel_dim, look_back_channel_dim
                    )
                )
                for _ in range(n_states)
            ]
        )

    def get_transition_matrix(self):
        return F.softmax(self.transition_matrix, dim=1)  # Each row sums to 1

    def get_initial_distribution(self):
        return F.softmax(self.initial_distribution, dim=0)

    def get_covariance_matrices(self):
        """Get positive definite covariance matrices using Cholesky decomposition"""
        covars = []
        for chol in self.covar_chol:
            # Make sure diagonal is positive
            chol_tril = torch.tril(chol)
            diag_indices = torch.arange(chol_tril.size(0))
            chol_tril[diag_indices, diag_indices] = torch.exp(
                chol_tril[diag_indices, diag_indices]
            )
            covar = chol_tril @ chol_tril.T
            covars.append(covar)
        return covars

    def get_gaussian_log_likelihood(self, emissions: Tensor):
        """
        Compute log likelihood of emissions under multivariate Gaussian for each state
        emissions.shape == (B, L, C)
        Returns: (B, L, n_states)
        """

        B, L, C = emissions.shape

        padded = torch.cat(
            [torch.zeros_like(emissions), emissions], dim=1
        )  # (B, 2*L, C)
        inputs = padded.unfold(dimension=1, size=L, step=1)  # (B, F, L, C)
        import pdb 
        pdb.set_trace()
        inputs = inputs[:, :L, :, :]
        inputs = rearrange(inputs, "B F C L -> (B F) L C")  # (B*L, L, C)
        targets = rearrange(emissions, "B L C -> (B L) C")  # (B*L, C)

        log_probs = torch.zeros(B, L, self.n_states, device=emissions.device)

        covars = self.get_covariance_matrices()

        for state in range(self.n_states):
            hidden_state_mean = (
                self.means[state].unsqueeze(0).expand(B * L, -1)
            )  # (B,C)
            covar = covars[state]  # (C, C)
            covar_reg = covar + 1e-6 * torch.eye(C, device=covar.device).unsqueeze(
                0
            ).expand(B * L, -1, -1)
            # add linear autoregressive compontent b=B*L d=C
            ar_component = torch.einsum(
                "blc,blcd->bd",
                inputs,
                self.W[state].unsqueeze(0).expand(B * L, -1, -1, -1),
            )

            mean = hidden_state_mean + ar_component

            dist = torch.distributions.MultivariateNormal(mean, covar_reg)
            log_prob = dist.log_prob(targets)  # (B*L, )
            log_prob = rearrange(log_prob, "(B L) -> B L", L=L)
            log_probs[:, :, state] = log_prob

        return log_probs

    def forward_algorithm(self, emissions: Tensor):
        """
        Forward algorithm to compute log likelihood
        emissions.shape == (B, L, C)
        Returns: log_likelihood (B,)
        """
        B, L, C = emissions.shape

        # Get emission probabilities
        emission_log_probs = self.get_gaussian_log_likelihood(
            emissions
        )  # (B, L, n_states)

        # Get HMM parameters
        init_dist = self.get_initial_distribution()  # (n_states,)
        trans_matrix = self.get_transition_matrix()  # (n_states, n_states)

        # Initialize forward probabilities
        log_alpha = torch.full(
            (B, L, self.n_states), -float("inf"), device=emissions.device
        )

        # Initial step
        log_alpha[:, 0, :] = torch.log(init_dist) + emission_log_probs[:, 0, :]

        # Forward pass
        for t in range(1, L):
            for j in range(self.n_states):
                # log_alpha[t, j] = log(sum_i(alpha[t-1, i] * trans[i, j])) + emission[t, j]
                log_trans = torch.log(trans_matrix[:, j] + 1e-8)  # (n_states,)
                log_alpha[:, t, j] = (
                    torch.logsumexp(
                        log_alpha[:, t - 1, :] + log_trans.unsqueeze(0), dim=1
                    )
                    + emission_log_probs[:, t, j]
                )

        # Final log likelihood
        log_likelihood = torch.logsumexp(log_alpha[:, -1, :], dim=1)  # (B,)
        return log_likelihood, log_alpha

    def forward(self, emissions: Tensor) -> Tensor:
        """
        Generate predictions using the most likely state sequence
        emissions.shape == (B, L, C)
        Returns: predictions (B, prediction_window, C)
        """
        B, L, C = emissions.shape

        # Get most likely state sequence for the look_back_window
        # hmm_emissions = emissions[:, -self.look_back_window :, :]
        # most_likely_states = self.viterbi(hmm_emissions)  # (B, look_back_window)
        _, log_alpha = self.forward_algorithm(emissions)
        most_likely_state = log_alpha[:, -1, :].argmax(dim=1)  # (B,) - last state

        # Get transition matrix
        trans_matrix = self.get_transition_matrix()  # (n_states, n_states)

        preds = []
        current_emissions = emissions.clone()

        for _ in range(self.prediction_window):
            # Predict next state
            next_state_dist = trans_matrix[most_likely_state, :]  # (B, n_states)
            most_likely_state = torch.argmax(next_state_dist, dim=1)  # (B,)

            # Generate prediction using autoregressive component
            look_back_data = current_emissions[
                :, -self.look_back_window :, :
            ]  # (B, L, C)

            ar_W = torch.stack(
                [self.W[i] for i in range(self.n_states)], dim=0
            )  # (n_state, L, C, C)
            W = ar_W[most_likely_state]  # (B, L, C, C)

            ar_pred = torch.einsum("blc,blcd->bd", look_back_data, W)  # (B,C)

            # Add state-specific mean
            state_means = torch.stack(
                [self.means[i] for i in range(self.n_states)], dim=0
            )  # (n_states, C)
            state_mean = state_means[most_likely_state]  # (B, C)

            current_pred = ar_pred + state_mean  # (B, C)
            preds.append(current_pred.unsqueeze(1))  # (B, 1, C)

            # Update emissions for next prediction
            current_emissions = torch.cat(
                [
                    current_emissions[:, -(self.look_back_window - 1) :, :],
                    current_pred.unsqueeze(1),
                ],
                dim=1,
            )

        preds = torch.cat(preds, dim=1)  # (B, prediction_window, C)
        return preds


class HMM(BaseLightningModule):
    def __init__(
        self,
        model: LinearAutoregressiveHMM,
        learning_rate: float = 1e-3,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate

    def model_forward(self, look_back_window: Tensor):
        preds = self.model(look_back_window)
        return preds

    def _shared_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss, _ = self.model.forward_algorithm(look_back_window)  # (B, )
        loss = -loss.mean()
        return loss

    def model_specific_train_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        loss = self._shared_step(look_back_window, prediction_window)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(
        self, look_back_window: Tensor, prediction_window: Tensor
    ) -> Tensor:
        val_loss = self._shared_step(look_back_window, prediction_window)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True)
        return val_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        return optimizer
