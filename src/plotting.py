import numpy as np
import torch
import wandb
import matplotlib.pyplot as plt
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as pcolors
import seaborn as sns

from lightning.pytorch.loggers import WandbLogger
from plotly.subplots import make_subplots


metric_names = ["MSE", "abs_target_mean", "cross_correlation"]
name_to_title = {
    "MSE": "MSE",
    "MAE": "MAE",
    "cross_correlation": "Pearson Correlation",
    "dir_acc_full": "Dir Acc Full",
    "dir_acc_single": "Dir Acc Single",
    "sMAPE": "sMAPE",
}

sns.set_theme(style="whitegrid")


def plot_entire_series(
    logger: WandbLogger,
    data: list[np.ndarray],
    metrics: dict,
    look_back_window: int,
    prediction_window: int,
):
    # n_metrics = len(metrics)
    n_metrics = 3  # only plot mse
    subplot_titles = ["Ground Truth Time Series"] + [
        name_to_title[metric_name] for metric_name in ["MSE", "MAE", "dir_acc_single"]
    ]
    colors = pcolors.qualitative.Plotly
    row_heights = [0.7] + [0.3 / n_metrics for _ in range(n_metrics)]

    window_length = look_back_window + prediction_window
    lengths = [len(s) - window_length + 1 for s in data]
    cum_lengths = np.cumsum([0] + lengths)

    assert cum_lengths[-1] == len(metrics["MSE"])

    n_series = len(data)
    for j in range(n_series):
        print(f"Plotting entire series {j + 1}")
        fig = make_subplots(
            rows=1 + n_metrics,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )
        series = data[j][:, 0]
        n = len(series)
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=series,
                mode="lines",
                name="Actual Value",
                line=dict(color=colors[0]),
            ),
            row=1,
            col=1,
        )

        for i, metric_name in enumerate(["MSE", "MAE", "dir_acc_single"]):
            mse_metric = np.array(
                metrics[metric_name][cum_lengths[j] : cum_lengths[j + 1]]
            )

            mse_plot = np.zeros(n)
            assert len(mse_metric) < len(mse_plot)
            mse_plot[look_back_window - 1 : look_back_window - 1 + len(mse_metric)] = (
                mse_metric
            )

            fig.add_trace(
                go.Bar(
                    x=list(range(n)),
                    y=mse_plot,
                    name=name_to_title[metric_name],
                    marker_color=colors[i + 1],
                    opacity=1.0,
                ),
                row=i + 2,
                col=1,
            )

        logger.experiment.log(
            {f"entire_series/series_vs_metric_{j}": wandb.Html(pio.to_html(fig))}
        )


def plot_metric_histogram(logger: WandbLogger, metric_name: str, data: list[float]):
    print(f"Plotting histogram for {metric_name}")
    fig, ax = plt.subplots(figsize=(8, 6))

    ax.hist(data, bins=50, edgecolor="black", alpha=0.7)  # Adjust bins as needed
    ax.set_title(f"Distribution of {name_to_title.get(metric_name, metric_name)}")
    ax.set_xlabel(name_to_title.get(metric_name, metric_name))
    ax.set_ylabel("Frequency")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    logger.experiment.log({f"{metric_name}/histogram": wandb.Image(fig)})
    plt.close(fig)


def get_yaxis_name(dataset: str, use_heart_rate: bool = False):
    if dataset in ["dalia", "wildppg", "ieee"]:
        yaxis_name = "Heartrate" if use_heart_rate else "PPG"
    elif dataset == "mhc6mwt":
        yaxis_name = "Heartrate"
    elif dataset in ["chapman", "ptbxl"]:
        yaxis_name = "ECG"
    elif dataset in ["ucihar", "usc"]:
        yaxis_name = "Acc & Gyro"
    elif dataset == "capture24":
        yaxis_name = "Acc"
    else:
        raise NotImplementedError()

    return yaxis_name


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
    dataset: str = "dalia",
    pred_denorm_std: torch.Tensor = None,
):
    # make sure to always plot only the first channel
    look_back_window = x.cpu().detach().numpy()[0, :, 0]
    target = y.cpu().detach().numpy()[0, :, 0]
    prediction = preds.cpu().detach().numpy()[0, :, 0]

    yaxis_name = get_yaxis_name(dataset, use_heart_rate)

    assert look_back_window.ndim == 1, f"look_back_window: {look_back_window.shape}"
    assert target.ndim == 1, f"target: {target.shape}"
    assert prediction.ndim == 1, f"prediction: {prediction.shape}"

    last_look_back = np.array([look_back_window[-1]])
    target = np.concatenate((last_look_back, target))
    prediction = np.concatenate((last_look_back, prediction))

    if use_heart_rate:
        t_lookback = np.arange(-(len(look_back_window) - 1), 1) * 2
        t_future = np.arange(0, len(target)) * 2
    else:
        t_lookback = np.linspace(
            -len(look_back_window) / freq, 0, len(look_back_window)
        )
        t_future = np.linspace(0, len(target) / freq, len(target))

    colors = sns.color_palette("colorblind", 3)
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.plot(t_lookback, look_back_window, color=colors[0], label="Lookback")
    ax.plot(t_future, target, color=colors[1], label="Ground Truth")
    ax.plot(t_future, prediction, color=colors[2], label="Prediction")

    if pred_denorm_std is not None:
        pred_denorm_std_np = pred_denorm_std.cpu().detach().numpy()[0, :, 0]

        std_full = np.concatenate(
            (np.array([pred_denorm_std_np[0]]), pred_denorm_std_np)
        )

        z_score = 1.96  # For 95% CI
        lower_bound = prediction - z_score * std_full
        upper_bound = prediction + z_score * std_full

        ax.fill_between(
            t_future,
            lower_bound,
            upper_bound,
            color=colors[2],
            alpha=0.2,
            label="95% Confidence Interval",
        )

    # Add a vertical line to mark the transition from look-back to prediction
    ax.axvline(x=0, color="gray", linestyle=":", linewidth=1, label="Forecast Start")

    title = f"{metric_name} {type} {metric_value:.3f}"
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Seconds", fontsize=12)
    ax.set_ylabel(yaxis_name, fontsize=12)

    ax.legend(loc="upper left", fontsize=10, frameon=True)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

    plt.tight_layout()

    fig.canvas.draw()
    print(f"Plotting example {metric_name}/{type}")
    wandb_logger.experiment.log({f"{metric_name}/{type}": wandb.Image(fig)})
    plt.close(fig)


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
        x, y, preds, None, "MSE", 0.1, "best", True, freq=32, dataset="ucihar"
    )
