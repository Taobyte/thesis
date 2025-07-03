import torch

from src.losses import get_loss_fn
from src.models.utils import BaseLightningModule


class Model(torch.nn.Module):
    def __init__(
        self,
        hidden_dim: int = 32,
        look_back_window: int = 5,
        prediction_window: int = 3,
        target_channel_dim: int = 1,
        look_back_channel_dim: int = 1,
        use_dynamic_features: bool = False,
        use_static_features: bool = False,
        dynamic_exogenous_variables: int = 1,
        static_exogenous_variables: int = 0,
    ):
        super().__init__()
        self.look_back_window = look_back_window
        self.prediction_window = prediction_window
        self.window_length = look_back_window + prediction_window

        self.target_channel_dim = target_channel_dim
        self.look_back_channel_dim = look_back_channel_dim
        self.use_dynamic_features = use_dynamic_features
        self.use_static_features = use_static_features
        self.dynamic_exogenous_variables = dynamic_exogenous_variables
        self.static_exogenous_variables = static_exogenous_variables

        self.base_dim = (
            hidden_dim  # latent space dimension without adding in additional info
        )

        if use_dynamic_features:
            hidden_dim += dynamic_exogenous_variables

        if use_static_features:
            hidden_dim += static_exogenous_variables
            self.static_start = (
                self.base_dim
                if not use_dynamic_features
                else self.base_dim + dynamic_exogenous_variables
            )

        # learnable transition matrices from motion model
        self.F = torch.nn.ModuleList(
            [torch.nn.Linear(hidden_dim, hidden_dim) for _ in range(self.window_length)]
        )
        # learnable observation matrices from sensor model
        self.H = torch.nn.ModuleList(
            [
                torch.nn.Linear(hidden_dim, target_channel_dim)
                for _ in range(self.window_length)
            ]
        )

        # learnable control input matrices
        if use_dynamic_features:
            self.B = torch.nn.ModuleList(
                [
                    torch.nn.Linear(dynamic_exogenous_variables, hidden_dim)
                    for _ in range(self.window_length)
                ]
            )

        # learnable initial state distribution
        self.initial_state = torch.nn.Parameter(torch.randn((hidden_dim,)))
        self.initial_covariance = torch.nn.Parameter(
            torch.randn((hidden_dim, hidden_dim))
        )

        # learnable noise matrices
        self.Q = torch.nn.Parameter(torch.randn(hidden_dim, hidden_dim))
        self.R = torch.nn.Parameter(torch.randn(target_channel_dim, target_channel_dim))

    def find_current_hidden_state(self, observation: torch.Tensor) -> torch.Tensor:
        """
        observation.shape == (B, T, self.look_back_channel_dim)
        Returns:
            hidden_state: (B, hidden_dim)
        """

        B, T, _ = observation.shape
        device = observation.device
        hidden_state = self.initial_state.unsqueeze(0).repeat(B, 1)  # (B, hidden_dim)

        P = self.initial_covariance.unsqueeze(0).repeat(
            B, 1, 1
        )  # (B, hidden_dim, hidden_dim)
        I = torch.eye(P.size(-1), device=device).unsqueeze(0).repeat(B, 1, 1)
        # pdb.set_trace()
        for t in range(self.look_back_window):
            F_t = self.F[t]
            H_t = self.H[t]
            if self.use_dynamic_features:
                B_t = self.B[t]
                dynamic_ex_var = observation[:, t, self.target_channel_dim :]

            # (S1) Prediction Step
            hidden_state_pred = F_t(hidden_state)  # (B, hidden_dim)
            if self.use_dynamic_features:
                hidden_state_pred += B_t(dynamic_ex_var)
            F_mat = F_t.weight  # (hidden_dim, hidden_dim)
            Q = self.Q.unsqueeze(0).repeat(B, 1, 1)
            P_pred = F_mat @ P @ F_mat.transpose(-1, -2) + Q

            # (S2) Measurement Update Step
            H_mat = H_t.weight  # (target_channel_dim, hidden_dim)
            R = self.R.unsqueeze(0).repeat(B, 1, 1)
            y_pred = H_t(hidden_state_pred)  # (B, target_channel_dim)
            y_obs = observation[
                :, t, : self.target_channel_dim
            ]  # (B, target_channel_dim)
            residual = y_obs - y_pred  # (B, target_channel_dim)

            H_mat_T = H_mat.transpose(-1, -2)
            S = H_mat @ P_pred @ H_mat_T + R
            K = P_pred @ H_mat_T @ torch.linalg.inv(S)  # Kalman Gain

            hidden_state = hidden_state_pred + (K @ residual.unsqueeze(-1)).squeeze(-1)
            P = (I - K @ H_mat) @ P_pred

        return hidden_state

    def forward(self, x: torch.Tensor):
        # x.shape == (B, T, 1)
        hidden_state = self.find_current_hidden_state(x)
        preds = []
        for i in range(self.look_back_window, self.window_length):
            hidden_state = self.F[i](hidden_state)
            emission = self.H[i](hidden_state)
            preds.append(emission)  # (B, 1, 1)

        stacked_preds = torch.cat(preds, dim=1)

        return stacked_preds.unsqueeze(-1)


class KalmanFilter(BaseLightningModule):
    def __init__(
        self,
        model: torch.nn.Module,
        loss: str = "MSE",
        learning_rate: float = 0.001,
        weight_decay: float = 0.0,
        exo_weight: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.model = model
        self.criterion = get_loss_fn(loss)
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.exo_weight = exo_weight

    def model_forward(self, look_back_window):
        assert len(look_back_window.shape) == 3

        # look_back_window = rearrange(look_back_window, "B T C -> B (T C)")
        preds = self.model(look_back_window)

        return preds[:, :, : self.model.target_channel_dim]

    def model_specific_train_step(self, look_back_window, prediction_window):
        preds = self.model(look_back_window)
        assert preds.shape == prediction_window.shape
        loss = self.criterion(preds, prediction_window)
        self.log("train_loss", loss, on_epoch=True, on_step=True, logger=True)

        return loss

    def model_specific_val_step(self, look_back_window, prediction_window):
        preds = self.model_forward(look_back_window)
        assert preds.shape == prediction_window.shape
        if self.tune:
            mae_criterion = torch.nn.L1Loss()
            loss = mae_criterion(preds, prediction_window)
        else:
            loss = self.criterion(preds, prediction_window)
        self.log("val_loss", loss, on_epoch=True, on_step=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        return optimizer
