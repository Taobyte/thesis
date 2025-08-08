import torch
import torch.nn.functional as F
from torch import Tensor
import math


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
        self.W = torch.nn.Parameter(
            torch.randn(look_back_window, look_back_channel_dim, prediction_window)
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
        log_probs = torch.zeros(B, L, self.n_states, device=emissions.device)

        covars = self.get_covariance_matrices()

        for state in range(self.n_states):
            mean = self.means[state]  # (C,)
            covar = covars[state]  # (C, C)

            # Compute multivariate Gaussian log probability
            # Add small epsilon to diagonal for numerical stability
            covar_reg = covar + 1e-6 * torch.eye(C, device=covar.device)

            try:
                chol = torch.linalg.cholesky(covar_reg)
                # Solve for (x - mu)
                diff = emissions - mean.unsqueeze(0).unsqueeze(0)  # (B, L, C)

                # Efficient computation using Cholesky decomposition
                solved = torch.triangular_solve(diff.unsqueeze(-1), chol, upper=False)[
                    0
                ]
                mahal_dist = (solved**2).sum(dim=-2).squeeze(-1)  # (B, L)

                log_det = 2 * torch.diagonal(chol, dim1=-2, dim2=-1).log().sum()
                log_prob = -0.5 * (mahal_dist + log_det + C * math.log(2 * math.pi))

            except RuntimeError:
                # Fallback if Cholesky fails
                covar_reg = covar + 1e-3 * torch.eye(C, device=covar.device)
                dist = torch.distributions.MultivariateNormal(mean, covar_reg)
                log_prob = dist.log_prob(emissions)  # (B, L)

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
        return log_likelihood

    def viterbi(self, emissions: Tensor):
        """
        Viterbi algorithm to find most likely state sequence
        emissions.shape == (B, L, C)
        Returns: most_likely_states (B, L)
        """
        B, L, C = emissions.shape

        # Get emission probabilities
        emission_log_probs = self.get_gaussian_log_likelihood(
            emissions
        )  # (B, L, n_states)

        # Get HMM parameters
        init_dist = self.get_initial_distribution()  # (n_states,)
        trans_matrix = self.get_transition_matrix()  # (n_states, n_states)

        # Viterbi tables
        log_delta = torch.full(
            (B, L, self.n_states), -float("inf"), device=emissions.device
        )
        psi = torch.zeros(
            (B, L, self.n_states), dtype=torch.long, device=emissions.device
        )

        # Initialize
        log_delta[:, 0, :] = torch.log(init_dist) + emission_log_probs[:, 0, :]

        # Forward pass
        for t in range(1, L):
            for j in range(self.n_states):
                # Find most likely previous state
                log_trans = torch.log(trans_matrix[:, j] + 1e-8)  # (n_states,)
                scores = log_delta[:, t - 1, :] + log_trans.unsqueeze(
                    0
                )  # (B, n_states)
                log_delta[:, t, j] = (
                    torch.max(scores, dim=1)[0] + emission_log_probs[:, t, j]
                )
                psi[:, t, j] = torch.argmax(scores, dim=1)

        # Backtrack
        states = torch.zeros((B, L), dtype=torch.long, device=emissions.device)
        states[:, -1] = torch.argmax(log_delta[:, -1, :], dim=1)

        for t in range(L - 2, -1, -1):
            for b in range(B):
                states[b, t] = psi[b, t + 1, states[b, t + 1]]

        return states

    def get_log_likelihood(self, emissions: Tensor):
        """
        Compute log likelihood for training
        emissions.shape == (B, L, C)
        """
        B, L, C = emissions.shape
        assert L > self.look_back_window, (
            f"Sequence length {L} must be > look_back_window {self.look_back_window}"
        )

        # Use only the look_back_window for HMM inference
        hmm_emissions = emissions[:, : self.look_back_window, :]
        return self.forward_algorithm(hmm_emissions)

    def forward(self, emissions: Tensor) -> Tensor:
        """
        Generate predictions using the most likely state sequence
        emissions.shape == (B, L, C)
        Returns: predictions (B, prediction_window, C)
        """
        B, L, C = emissions.shape

        # Get most likely state sequence for the look_back_window
        hmm_emissions = emissions[:, -self.look_back_window :, :]
        most_likely_states = self.viterbi(hmm_emissions)  # (B, look_back_window)
        most_likely_state = most_likely_states[:, -1]  # (B,) - last state

        # Get transition matrix
        trans_matrix = self.get_transition_matrix()  # (n_states, n_states)

        preds = []
        current_emissions = emissions.clone()

        for step in range(self.prediction_window):
            # Predict next state
            next_state_dist = trans_matrix[most_likely_state, :]  # (B, n_states)
            most_likely_state = torch.argmax(next_state_dist, dim=1)  # (B,)

            # Generate prediction using autoregressive component
            look_back_data = current_emissions[
                :, -self.look_back_window :, :
            ]  # (B, look_back_window, C)

            # Apply linear transformation: (B, look_back_window, C) @ (look_back_window, C, 1) -> (B, C)
            ar_pred = torch.einsum(
                "blc,lcw->bwc", look_back_data, self.W
            )  # (B, prediction_window=1, C)
            ar_pred = ar_pred.squeeze(1)  # (B, C)

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
