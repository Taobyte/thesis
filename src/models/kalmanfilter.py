import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd.functional as AF

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class TransitionModule(nn.Module):
    def reset_states(self, batch_size: int, device: torch.device):
        """
        Reset hidden state.

        Dummy implementation: no operation performed.
        """

    @torch.jit.ignore()
    def jacobian(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute Jacobian matrix of the transition function w.r.t. state

        Args:
            x: Current state [batch_size, state_dim]
            u: Control input [batch_size, control_dim] (optional)

        Returns:
            jacobian: Batch of Jacobian matrices [batch_size, state_dim, state_dim]
        """
        batch_size = x.shape[0]

        # Compute the Jacobian
        # This will return a tensor of shape [batch_size, state_dim, batch_size, state_dim]
        if u is not None and self.control_dim > 0:
            jacobians = AF.jacobian(
                lambda x_: self.forward(x_, u), x.detach(), create_graph=self.training
            )
        else:
            jacobians = AF.jacobian(
                self.forward, x.detach(), create_graph=self.training
            )

        # Extract the diagonal elements corresponding to each batch item
        batch_indices = torch.arange(batch_size, device=x.device)
        jacobians = jacobians[batch_indices, :, batch_indices, :]

        return jacobians


class LinearTransitionModule(TransitionModule):
    """Linear backbone: x_{t+1} = Ax_t + Bu_t"""

    def __init__(
        self,
        state_dim: int,
        control_dim: int,
        learnable: bool = True,
        explicit_covariance: bool = False,
        initial_covariance: float = 1e-3,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.control_dim = control_dim

        # Initialize transition matrix A
        A = torch.eye(state_dim) + 0.01 * torch.randn(state_dim, state_dim)
        if learnable:
            self.transition_matrix = nn.Parameter(A)
        else:
            self.register_buffer("A", A)

        # Initialize control matrix B (if control_dim > 0)
        if control_dim > 0:
            B = 0.01 * torch.randn(state_dim, control_dim)
            if learnable:
                self.control_matrix = nn.Parameter(B)
            else:
                self.register_buffer("B", B)
        else:
            self.control_matrix = None

        is_learnable_cov = explicit_covariance and control_dim > 0
        log_process_noise = torch.ones(state_dim) * torch.log(
            torch.tensor(initial_covariance)
        )
        if is_learnable_cov:
            self.log_process_noise = nn.Parameter(log_process_noise)
        else:
            self.register_buffer("log_process_noise", log_process_noise)
        # logging.getLogger(__name__).info(
        #     f"{self.__class__.__name__}: log_process_noise learnable: {is_learnable_cov}"
        # )

    def forward(
        self, state: torch.Tensor, control: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        next_state = torch.matmul(state, self.transition_matrix.t())

        if control is not None and self.control_matrix is not None:
            next_state = next_state + torch.matmul(control, self.control_matrix.t())

        return next_state

    def jacobian(
        self, x: torch.Tensor, u: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return transition matrix as the Jacobian

        Args:
            x: Current state [batch_size, state_dim]
            u: Control input [batch_size, control_dim] (optional)

        Returns:
            jacobian: Batch of Jacobian matrices [batch_size, state_dim, state_dim]
        """
        batch_size = x.shape[0]
        # Expand transition matrix to batch dimension
        return self.transition_matrix.expand(batch_size, self.state_dim, self.state_dim)

    def covariance(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Return process noise covariance matrix

        Args:
            batch_size: Batch size
            device: Device to create tensor on

        Returns:
            Q: Process noise covariance [batch_size, state_dim, state_dim]
        """
        cov_diag = torch.exp(self.log_process_noise)
        cov_matrix = torch.diag(cov_diag).to(device)
        return cov_matrix.expand(batch_size, self.state_dim, self.state_dim)


class BaseInnovationModule(nn.Module):
    def __init__(self, obs_dim: int):
        super(BaseInnovationModule, self).__init__()
        self.obs_dim = obs_dim

    def forward(
        self, observation: torch.Tensor, pred_observation: torch.Tensor
    ) -> torch.Tensor:
        raise NotImplementedError


class ClassicalInnovationModule(BaseInnovationModule):
    def forward(
        self, observation: torch.Tensor, pred_observation: torch.Tensor
    ) -> torch.Tensor:
        return observation - pred_observation


class ObservationModule(nn.Module):
    """Base class for observation models"""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim

    def reset_states(self, batch_size: int, device: torch.device):
        """Reset hidden state. Dummy implementation: no operation performed."""
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute observation from state"""
        raise NotImplementedError

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        """Compute Jacobian matrix of the observation function w.r.t. state"""
        batch_size = x.shape[0]
        jacobians = torch.autograd.functional.jacobian(
            self.forward, x.detach(), create_graph=self.training
        )
        batch_indices = torch.arange(batch_size, device=x.device)
        jacobians = jacobians[batch_indices, :, batch_indices, :]
        return jacobians


class LinearObservationModule(ObservationModule):
    """Linear observation model: y_k = Cx_k + v_k"""

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        learnable: bool = True,
        explicit_covariance: bool = False,
        initial_covariance: float = 1e-3,
    ):
        super().__init__(state_dim, obs_dim)

        observation_matrix = torch.eye(obs_dim, state_dim)
        if learnable:
            self.observation_matrix = nn.Parameter(observation_matrix)
        else:
            self.register_buffer("observation_matrix", observation_matrix)

        # Observation noise covariance
        log_observation_cov = torch.ones(obs_dim) * torch.log(
            torch.tensor(initial_covariance)
        )
        if explicit_covariance:
            self.register_buffer("log_observation_cov", log_observation_cov)
        else:
            self.log_observation_cov = nn.Parameter(log_observation_cov)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(x, self.observation_matrix.t())

    def jacobian(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        return self.observation_matrix.expand(batch_size, self.obs_dim, self.state_dim)

    def covariance(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Return observation noise covariance matrix"""
        cov_diag = torch.exp(self.log_observation_cov)
        cov_matrix = torch.diag(cov_diag).to(device)
        return cov_matrix.expand(batch_size, self.obs_dim, self.obs_dim)


class BaseKalmanFilter(nn.Module, ABC):
    """
    Extended base class for traditional Kalman filter variants that use
    modular transition/observation/innovation components.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        control_dim: int,
    ):
        super().__init__()
        self.state_dim = state_dim
        self.obs_dim = obs_dim
        self.control_dim = control_dim

        # Register initial state
        self.register_buffer("initial_state", torch.zeros(self.state_dim))

    @abstractmethod
    def predict(
        self,
        observations: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: encode -> decode.

        Args:
            observations: [B, T, obs_dim]
            controls: [B, T, control_dim] or None
            initial_state: [B, state_dim] or None

        Returns:
            Dictionary containing:
                - "recon_observations": [B, T, obs_dim]
                - "filtered_states": [B, T, state_dim]
                - Additional filter-specific outputs
        """

    @abstractmethod
    def compute_losses(
        self,
        observations: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        true_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute all losses for training.

        Args:
            observations: [B, T, obs_dim]
            controls: [B, T, control_dim] or None
            true_states: [B, T, state_dim] or None for supervision

        Returns:
            Dictionary of loss components including "total"
        """

    def get_current_params(self) -> Dict[str, Any]:
        """
        Returns a dictionary of key model parameters useful for monitoring.
        """
        return {
            "state_dim": self.state_dim,
            "obs_dim": self.obs_dim,
            "control_dim": self.control_dim,
        }


class Model(BaseKalmanFilter):
    """
    Classical Kalman Filter with unified API matching dkf.py interface.
    """

    def __init__(
        self,
        state_dim: int,
        obs_dim: int,
        smoothness_weight: float = 0.01,
        reconstruction_weight: float = 0.00,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        dynamic_exogenous_variables: int = 0,
        static_exogenous_variables: int = 0,
        target_channel_dim: int = 1,
        look_back_window: int = 5,
        prediction_window: int = 3,
    ):
        control_dim = 0
        if use_dynamic_features:
            control_dim += dynamic_exogenous_variables
        if use_static_features:
            control_dim += static_exogenous_variables

        super().__init__(state_dim=state_dim, obs_dim=obs_dim, control_dim=control_dim)

        self.transition_model = LinearTransitionModule(state_dim, control_dim)
        self.observation_model = LinearObservationModule(state_dim, obs_dim)
        self.innovation_model = ClassicalInnovationModule(obs_dim)
        self.smoothness_weight = smoothness_weight
        self.reconstruction_weight = reconstruction_weight

        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features

        self.target_channel_dim = target_channel_dim

        self.look_back_window = look_back_window
        self.prediction_window = prediction_window

    def _prediction_step(
        self,
        prev_state: torch.Tensor,
        prev_covariance: torch.Tensor,
        control: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Prediction step of the Kalman filter.

        Args:
            prev_state: Previous state estimate.
            prev_covariance: Previous state covariance.
            control: Control input.

        Returns:
            Prior state estimate and prior state covariance.
        """
        batch_size = prev_state.shape[0]
        device = prev_state.device

        # Predict state using dynamics model
        prior_state = self.transition_model(prev_state, control)

        # Get Jacobian of dynamics function
        F = self.transition_model.jacobian(prev_state, control)

        # Get process noise covariance
        Q = self.transition_model.covariance(batch_size, device)

        # Update covariance
        prior_covariance = (
            torch.bmm(torch.bmm(F, prev_covariance), F.transpose(1, 2)) + Q
        )

        return prior_state, prior_covariance

    def _kgain_step(
        self,
        prior_covariance: torch.Tensor,
        observation_matrix: torch.Tensor,
        observation_noise: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the Kalman gain.

        Args:
            prior_covariance: Prior state covariance.
            observation_matrix: Observation matrix.
            observation_noise: Observation noise covariance.

        Returns:
            Kalman gain.
        """
        device = prior_covariance.device

        # Compute innovation covariance
        S = (
            torch.bmm(
                torch.bmm(observation_matrix, prior_covariance),
                observation_matrix.transpose(1, 2),
            )
            + observation_noise
        )

        S = S + torch.eye(S.size(1), device=device) * 1e-6
        # Compute Kalman gain
        K = torch.bmm(
            torch.bmm(prior_covariance, observation_matrix.transpose(1, 2)),
            torch.inverse(S),
        )

        return K

    def _update_step(
        self,
        prior_state: torch.Tensor,
        prior_covariance: torch.Tensor,
        observation: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update step of the Kalman filter.

        Args:
            prior_state: Prior state.
            prior_covariance: Prior state covariance.
            observation: Observation.
            innovation: Innovation.

        Returns:
            Posterior state and covariance.
        """
        batch_size = prior_state.shape[0]
        device = prior_state.device

        # Get Jacobian of observation function
        H = self.observation_model.jacobian(prior_state)

        # Get observation noise covariance
        R = self.observation_model.covariance(batch_size, device)

        # Compute the kalman gain
        K = self._kgain_step(prior_covariance, H, R)

        # Compute predicted observation and innovation
        pred_observation = self.observation_model(prior_state)
        innovation = self.innovation_model(observation, pred_observation)

        # Update state
        posterior_state = prior_state + torch.bmm(K, innovation.unsqueeze(-1)).squeeze(
            -1
        )

        # Get Jacobian of observation function
        H = self.observation_model.jacobian(prior_state)

        # Update covariance
        I = torch.eye(self.state_dim, device=device).expand(batch_size, -1, -1)
        posterior_covariance = torch.bmm((I - torch.bmm(K, H)), prior_covariance)

        return posterior_state, posterior_covariance, pred_observation

    def predict(
        self,
        observations: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Encode observations to filtered latent states using Kalman filtering.

        Args:
            observations: Tensor of shape (batch_size, seq_len, obs_dim).
            controls: Tensor of shape (batch_size, seq_len, control_dim).
            initial_state: Tensor of shape (batch_size, state_dim).

        Returns:
            Tensor of shape (batch_size, seq_len, state_dim).
        """

        batch_size, seq_len, _ = observations.shape
        device = observations.device

        # Initialize state and covariance
        if initial_state is None:
            state_estimate = self.initial_state.expand(batch_size, -1).to(device)
        else:
            state_estimate = initial_state

        state_covariance = (
            torch.eye(self.state_dim, device=device).expand(batch_size, -1, -1) * 1e-3
        )

        # Storage for results
        filtered_states = torch.zeros(
            batch_size, seq_len, self.state_dim, device=device
        )

        predicted_observations = torch.zeros(
            batch_size, seq_len, self.obs_dim, device=device
        )

        # Process sequence
        for t in range(seq_len):
            observation = observations[:, t]
            control = controls[:, t] if controls is not None else None

            # Prediction step
            prior_state, prior_covariance = self._prediction_step(
                state_estimate, state_covariance, control
            )

            # Update step
            posterior_state, posterior_covariance, pred_observations = (
                self._update_step(prior_state, prior_covariance, observation)
            )

            # Store results
            state_estimate = posterior_state
            state_covariance = posterior_covariance
            filtered_states[:, t] = state_estimate
            predicted_observations[:, t] = pred_observations

        return {
            "filtered_states": filtered_states,
            "recon_observations": predicted_observations,
        }

    def compute_losses(
        self,
        observations: torch.Tensor,
        controls: Optional[torch.Tensor] = None,
        true_states: Optional[torch.Tensor] = None,
        initial_state: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute Classical Kalman Filter losses.
        """
        # Get filtered states and predicted observations
        result = self.predict(observations, controls, initial_state)
        filtered_states = result["filtered_states"]
        pred_observations = result["recon_observations"]

        # Compute base losses
        losses: dict[str, torch.Tensor] = {}

        # Primary loss
        if true_states is not None:
            state_loss = F.mse_loss(filtered_states, true_states)
            losses["state_loss"] = state_loss
            losses["total"] = state_loss
        else:
            recon_loss = F.mse_loss(pred_observations, observations)
            mae_loss = F.l1_loss(pred_observations, observations)
            losses["reconstruction_loss"] = recon_loss
            losses["mae_loss"] = mae_loss
            losses["total"] = recon_loss

        # Smoothness regularization
        if filtered_states.shape[1] > 1 and self.smoothness_weight > 0:
            smoothness_loss = F.mse_loss(
                filtered_states[:, 1:], filtered_states[:, :-1]
            )
            losses["smoothness_loss"] = smoothness_loss * self.smoothness_weight
            losses["total"] = losses["total"] + losses["smoothness_loss"]

        return losses

    def forward(self, lookback_seq: torch.Tensor) -> torch.Tensor:
        if self.control_dim > 0:
            controls = lookback_seq[:, :, self.target_channel_dim :]
        else:
            controls = None

        prediction_dict = self.predict(lookback_seq, controls)

        current_state = prediction_dict["filtered_states"][:, -1, :]
        prediction = lookback_seq[:, -1, :].unsqueeze(1)

        preds: list[torch.Tensor] = []
        for _ in range(self.prediction_window):
            if self.control_dim > 0:
                last_pred_control = prediction[:, 0, self.target_channel_dim :]
            else:
                last_pred_control = None
            current_state = self.transition_model(current_state, last_pred_control)
            prediction = self.observation_model(current_state).unsqueeze(1)
            preds.append(prediction)

        concat_preds = torch.concat(preds, dim=1)

        return concat_preds


class KalmanFilter(BaseLightningModule):
    def __init__(
        self,
        model: BaseKalmanFilter,
        loss: str = "MSE",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        optimizer_name: str = "adam",
        **kwargs: Any,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.optimizer_name = optimizer_name

    def model_specific_forward(self, look_back_window: torch.Tensor):
        assert len(look_back_window.shape) == 3

        # look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        preds = self.model(look_back_window)

        return preds[:, :, : self.model.target_channel_dim]

    # def _shared_step(self, series: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    #     heartrate = series[:, :, : self.target_channel_dim]
    #     if self.model.control_dim > 0:
    #         activity = series[:, :, self.target_channel_dim :]
    #     else:
    #         activity = None

    #     losses = self.model.compute_losses(heartrate, activity)
    #     loss = losses["total"]
    #     mae_loss = losses["mae_loss"]
    #     return loss, mae_loss

    # def model_specific_train_step(self, series: torch.Tensor):
    #     loss, _ = self._shared_step(series)
    #     self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)
    #     return loss

    # def model_specific_val_step(self, series: torch.Tensor):
    #     val_loss, mae_loss = self._shared_step(series)
    #     loss = val_loss
    #     if self.tune:
    #         loss = mae_loss
    #     self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
    #     return loss

    def model_specific_train_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ):
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : self.target_channel_dim]
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        # loss = self.model.compute_losses(look_back_window, None, None)["total"]

        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)

        return loss

    def model_specific_val_step(
        self, look_back_window: torch.Tensor, prediction_window: torch.Tensor
    ):
        preds = self.model_forward(look_back_window)
        preds = preds[:, :, : self.target_channel_dim]
        assert preds.shape == prediction_window.shape
        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, prediction_window)
        else:
            loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
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
                self.model.parameters(), lr=self.learning_rate
            )
        return optimizer
