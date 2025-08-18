import torch
import torch.nn.functional as F

from einops import rearrange
from torch import Tensor
from typing import List, Any, Union

from src.models.utils import BaseLightningModule


class LinearAutoregressiveHMM(torch.nn.Module):
    def __init__(
        self,
        n_states: int = 2,
        look_back_window: int = 5,
        prediction_window: int = 3,
        look_back_channel_dim: int = 2,
        target_channel_dim: int = 1,
    ):
        super().__init__()
        self.n_states = n_states
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.look_back_channel_dim = look_back_channel_dim
        self.target_channel_dim = target_channel_dim

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
                torch.nn.Parameter(torch.randn((look_back_channel_dim,)))
                for _ in range(n_states)
            ]
        )

        # Autoregressive weights
        self.W = torch.nn.ModuleList(
            [
                torch.nn.Linear(
                    look_back_window * look_back_channel_dim,
                    look_back_channel_dim,
                    bias=False,
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
        covars = [torch.diag(torch.exp(v)) for v in self.covar_chol]
        #  for chol in self.covar_chol:
        #      # Make sure diagonal is positive
        #      chol_tril = torch.tril(chol)
        #      diag_indices = torch.arange(chol_tril.size(0))
        #      chol_tril[diag_indices, diag_indices] = torch.exp(
        #          chol_tril[diag_indices, diag_indices]
        #      )
        #      covar = chol_tril @ chol_tril.T
        #      covars.append(covar)
        return covars

    def get_inputs(self, emissions: Tensor, is_series: bool = True):
        device = emissions.device
        if is_series:
            B, S, C = emissions.shape
            pad = torch.zeros((B, self.look_back_window, C), device=device)
            concatenated = torch.concatenate([pad, emissions], dim=1)  # (B, L + S, C)
            inputs = concatenated.unfold(
                dimension=1, size=self.look_back_window, step=1
            )
            inputs = inputs[:, :-1, :, :]
            inputs = rearrange(inputs, "B F C L -> (B F) (C L)")

        else:
            padded = torch.cat(
                [torch.zeros_like(emissions, device=device), emissions], dim=1
            )  # (B, 2*L, C)
            inputs = padded.unfold(
                dimension=1, size=self.look_back_window, step=1
            )  # (B, F, L, C)
            inputs = inputs[:, : self.look_back_window, :, :]
            inputs = rearrange(inputs, "B F C L -> (B F) (L C)")  # (B*L, L*C)

        return inputs

    def get_gaussian_log_likelihood(self, emissions: Tensor, is_series: bool = False):
        """
        Compute log likelihood of emissions under multivariate Gaussian for each state
        emissions.shape == (B, L, C)
        Returns: (B, L, n_states)
        """

        B, L, C = emissions.shape

        inputs = self.get_inputs(emissions, is_series=is_series)
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
            ar_component = self.W[state](inputs)  # (B*L, C)
            mean = hidden_state_mean + ar_component

            dist = torch.distributions.MultivariateNormal(mean, covar_reg)
            log_prob = dist.log_prob(targets)  # (B*L, )
            log_prob = rearrange(log_prob, "(B L) -> B L", L=L)
            log_probs[:, :, state] = log_prob

        return log_probs

    def forward_algorithm_series(self, emissions: Tensor, lengths: Tensor):
        """
        emissions: (B, S, C)
        returns: (log_likelihood: (B,), log_alpha: (B, L_eff, K))
        """

        B, S, C = emissions.shape
        device = emissions.device
        K = self.n_states

        emission_log_probs = self.get_gaussian_log_likelihood(
            emissions, is_series=True
        )  # (B, S, K)

        init_dist = self.get_initial_distribution()  # (K,)
        trans = self.get_transition_matrix()  # (K, K)

        log_alpha = emissions.new_full((B, S, K), -float("inf"), device=device)

        log_alpha[:, 0, :] = torch.log(init_dist + 1e-12) + emission_log_probs[:, 0, :]

        # recurrence
        log_trans = torch.log(trans + 1e-12)  # (K, K)
        for t in range(1, S):
            # logsumexp over previous states i
            prev = log_alpha[:, t - 1, :].unsqueeze(2) + log_trans.unsqueeze(
                0
            )  # (B,K,K)
            log_alpha[:, t, :] = (
                torch.logsumexp(prev, dim=1) + emission_log_probs[:, t, :]
            )
        # log_likelihood = torch.logsumexp(log_alpha[:, -1, :], dim=1)
        # return log_likelihood, log_alpha
        last_idx = lengths - 1
        last_alpha = log_alpha[torch.arange(B, device=device), last_idx, :]  # (B, K)
        log_likelihood = torch.logsumexp(last_alpha, dim=1)  # (B,)

        return log_likelihood, log_alpha

    def forward_algorithm(self, emissions: Tensor):
        """
        Forward algorithm to compute log likelihood
        emissions.shape == (B, L, C)
        Returns: log_likelihood (B,)
        """

        B, L, C = emissions.shape

        assert L == self.look_back_window

        # Get emission probabilities
        emission_log_probs = self.get_gaussian_log_likelihood(
            emissions, is_series=False
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
            look_back_data_reshaped = rearrange(look_back_data, "B L C -> B (L C)")

            ar_pred_list: List[Tensor] = []
            for i, state in enumerate(most_likely_state):
                ar_pred_list.append(self.W[state](look_back_data_reshaped[i]))
            ar_pred = torch.stack(ar_pred_list, dim=0)
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
        optimizer_name: str = "lbfgs",
        weight_decay: float = 0.0,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.model = model
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name

        self.weight_decay = weight_decay

    def model_forward(self, look_back_window: Tensor):
        preds = self.model(look_back_window)
        return preds

    def _shared_step(self, series: Tensor, lengths: Tensor) -> Tensor:
        loss, _ = self.model.forward_algorithm_series(series, lengths)  # (B, )
        loss = -loss.mean()
        # criterion = torch.nn.MSELoss()
        # preds = self.model(look_back_window)
        # assert preds.shape == prediction_window.shape
        # loss =  criterion(preds, prediction_window)
        return loss

    def model_specific_train_step(self, series: Tensor, lengths: Tensor) -> Tensor:
        loss = self._shared_step(series, lengths)
        self.log("train_loss", loss, on_step=True, on_epoch=True, logger=True)
        return loss

    def model_specific_val_step(self, series: Tensor, lengths: Tensor) -> Tensor:
        val_loss = self._shared_step(series, lengths)
        self.log("val_loss", val_loss, on_step=True, on_epoch=True, logger=True)
        return val_loss

    def configure_optimizers(self) -> Union[torch.optim.Adam, torch.optim.LBFGS]:
        if self.optimizer_name == "adam":
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay,
            )
        elif self.optimizer_name == "lbfgs":
            optimizer = torch.optim.LBFGS(
                self.model.parameters(),
                lr=self.learning_rate,
            )
        else:
            raise NotImplementedError()

        return optimizer
