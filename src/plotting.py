import numpy as np
import torch
import wandb
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from lightning.pytorch.loggers import WandbLogger


sns.set_theme(style="whitegrid")

metric_names = ["MSE", "abs_target_mean", "cross_correlation"]
name_to_title = {"MSE": "MSE", "MAE": "MAE", "cross_correlation": "Pearson"}


def plot_ploty(
    t_lookback,
    t_future,
    look_back_window,
    target,
    prediction,
    metric_name,
    metric_value,
    type,
    yaxis_name,
    wandb_logger,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=t_lookback,
            y=look_back_window,
            mode="lines",
            name="Lookback",
            line=dict(color="#0000ff"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_future,
            y=target,
            mode="lines",
            name="Ground Truth",
            line=dict(color="#ffa000"),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=t_future,
            y=prediction,
            mode="lines",
            name="Prediction",
            line=dict(
                color="#ff0069",
            ),
        )
    )

    title = f"{name_to_title[metric_name]}: {metric_value}"

    fig.update_layout(
        title=title,
        xaxis_title="Seconds",
        yaxis_title=yaxis_name,
        legend=dict(x=0.01, y=0.99, borderwidth=1),
    )

    wandb_logger.experiment.log({f"{metric_name}/{type}": wandb.Plotly(fig)})


def plot_prediction_wandb(
    x: torch.Tensor,
    y: torch.Tensor,
    preds: torch.Tensor,
    wandb_logger: WandbLogger,
    metric_name: str,
    metric_value: float,
    type: str,
    use_heart_rate: bool = False,
    freq: int = 25,
    yaxis_name: str = "Heartrate",
):
    # make sure to always plot only the first channel
    look_back_window = x.cpu().detach().numpy()[0, :, 0]
    target = y.cpu().detach().numpy()[0, :, 0]
    prediction = preds.cpu().detach().numpy()[0, :, 0]

    assert look_back_window.ndim == 1, f"look_back_window: {look_back_window.shape}"
    assert target.ndim == 1, f"target: {target.shape}"
    assert prediction.ndim == 1, f"prediction: {prediction.shape}"

    if use_heart_rate:
        t_lookback = np.arange(-(len(look_back_window) - 1), 1) * 2
        t_future = np.arange(0, len(target) + 1) * 2
    else:
        t_lookback = np.linspace(
            -len(look_back_window) // freq, 0, len(look_back_window)
        )
        t_future = np.linspace(0, len(target) // freq, len(target))
    last_look_back = np.array([look_back_window[-1]])
    target = np.concatenate((last_look_back, target))
    prediction = np.concatenate((last_look_back, prediction))

    colors = sns.color_palette("colorblind", 3)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t_lookback, look_back_window, color=colors[0], label="Lookback")
    ax.plot(t_future, target, color=colors[1], label="Ground Truth")
    ax.plot(t_future, prediction, color=colors[2], label="Prediction")

    title = f"{metric_name} {type} {metric_value:.3f}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Seconds", fontsize=12)
    ax.set_ylabel(yaxis_name, fontsize=12)

    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    # fig.show()

    wandb_logger.experiment.log({f"{metric_name}/{type}": wandb.Image(fig)})


if __name__ == "__main__":
    # test plotting functionality
    x = torch.randn((1, 96, 2)).abs() + 100
    y = torch.randn((1, 32, 2)).abs() + 100
    preds = torch.randn((1, 32, 2)).abs() + 100

    metrics = {
        "cross_correlation": [0.1, 0.7, 0.3],
        "MSE": [0.9, 3.3, 11],
        "abs_target_mean": [0.3, 2, 0.87],
    }

    plot_prediction_wandb(
        x, y, preds, None, "MSE", 0.1, "best", True, freq=32, yaxis_name="PPG"
    )
