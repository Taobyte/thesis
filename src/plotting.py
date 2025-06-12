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
    datamodule,
    metrics: dict,
) -> None:
    """
    Plots each entire time series along with corresponding per-step evaluation metrics
    (MSE, MAE, Directional Accuracy) as bar plots.

    This function iterates through multiple time series, creating a multi-panel plot
    for each: the top panel displays the ground truth time series, and the panels below
    show the MSE, MAE, and Directional Accuracy metrics as bar plots aligned with the
    time series. The metrics are plotted at the time steps corresponding to the end
    of their respective lookback windows.

    Args:
        logger (WandbLogger): The Weights & Biases logger instance to log the generated plots.
        data (list[np.ndarray]): A list of NumPy arrays, where each array represents
                                 an entire time series. Each array is expected to have
                                 shape (sequence_length, num_channels), with the primary
                                 series to plot being in the first channel (index 0).
        metrics (dict): A dictionary containing evaluation metrics. Expected keys are
                        "MSE", "MAE", and "dir_acc_single". Each value is a list or
                        NumPy array of metric values, concatenated across all series
                        and time windows.
        look_back_window (int): The length of the look-back (input) window used by the model.
                                This is used to correctly offset the metric plots.
        prediction_window (int): The length of the prediction (output) window used by the model.
                                 This is used to calculate the total window length for alignment.
    """

    data = datamodule.test_dataset.data
    look_back_window = datamodule.look_back_window
    prediction_window = datamodule.prediction_window
    use_dynamic_features = datamodule.use_dynamic_features
    target_channel_dim = datamodule.target_channel_dim

    required_keys = {"MSE", "MAE", "dir_acc_single"}
    assert required_keys.issubset(metrics.keys())
    n_metrics = 3  # only plot mse, mae and dir acc
    series_title = (
        ["Ground Truth Time Series", "Activity Info"]
        if use_dynamic_features
        else ["Ground Truth Time Series"]
    )
    subplot_titles = series_title + [
        name_to_title[metric_name] for metric_name in ["MSE", "MAE", "dir_acc_single"]
    ]
    colors = pcolors.qualitative.Plotly
    series_heights = [0.35, 0.35] if use_dynamic_features else [0.7]
    row_heights = series_heights + [0.3 / n_metrics for _ in range(n_metrics)]

    window_length = look_back_window + prediction_window
    lengths = [len(s) - window_length + 1 for s in data]
    cum_lengths = np.cumsum([0] + lengths)

    assert cum_lengths[-1] == len(metrics["MSE"])

    n_series = len(data)
    for j in range(n_series):
        print(f"Plotting entire series {j + 1}")
        fig = make_subplots(
            rows=(2 if use_dynamic_features else 1) + n_metrics,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            row_heights=row_heights,
            subplot_titles=subplot_titles,
        )
        series = data[j][:, 0]
        activity = data[j][:, target_channel_dim]
        n = len(series)
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=series,
                mode="lines",
                name="Heartrate Value",
                line=dict(color=colors[0]),
            ),
            row=1,
            col=1,
        )
        if use_dynamic_features:
            fig.add_trace(
                go.Scatter(
                    x=list(range(n)),
                    y=activity,
                    mode="lines",
                    name="Activity Value",
                    line=dict(color=colors[1]),
                ),
                row=2,
                col=1,
            )
        offset = 1 if use_dynamic_features else 0
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
                    marker_color=colors[offset + i + 1],
                    opacity=1.0,
                ),
                row=offset + i + 2,
                col=1,
            )

        logger.experiment.log(
            {f"entire_series/series_vs_metric_{j}": wandb.Html(pio.to_html(fig))}
        )


def plot_metric_histograms(logger: WandbLogger, metric_full: dict[str]) -> None:
    """
    Generates and logs histograms for various evaluation metrics to Weights & Biases.

    For each metric provided in the `metric_full` dictionary, this function creates
    a histogram visualizing its distribution across all evaluated time windows.
    The plots are then logged as images to the specified WandbLogger instance.

    Args:
        logger (WandbLogger): The Weights & Biases logger instance to log the generated plots.
        metric_full (Dict[str, List[float]]): A dictionary where keys are metric names (str)
                                               and values are lists or NumPy arrays of
                                               all computed metric values across all series
                                               and time windows (e.g., `{"MSE": [0.1, 0.05, ...], "MAE": [0.2, 0.15, ...])`).
    """
    print("Plotting metric histograms")
    for metric_name, v in metric_full.items():
        fig, ax = plt.subplots(figsize=(8, 6))

        ax.hist(v, bins=50, edgecolor="black", alpha=0.7)  # Adjust bins as needed
        ax.set_title(f"Distribution of {name_to_title.get(metric_name, metric_name)}")
        ax.set_xlabel(name_to_title.get(metric_name, metric_name))
        ax.set_ylabel("Frequency")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.7)

        logger.experiment.log({f"{metric_name}/histogram": wandb.Image(fig)})
        plt.close(fig)


def get_yaxis_name(dataset: str, use_heart_rate: bool = False) -> str:
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
) -> None:
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


def plot_max_min_median_predictions(lightning_module) -> None:
    from src.models.utils import local_z_norm, local_z_denorm

    # plot best, worst and median
    for metric_name, v in lightning_module.metric_full.items():
        sorted_indices = np.argsort(v)
        min_idx = sorted_indices[0]
        max_idx = sorted_indices[-1]
        median_idx = sorted_indices[len(v) // 2]
        if metric_name == "cross_correlation":
            zipped = zip(["worst", "best", "median"], [min_idx, max_idx, median_idx])
        else:
            zipped = zip(["worst", "best", "median"], [max_idx, min_idx, median_idx])

        for type, idx in zipped:
            look_back_window, target = (
                lightning_module.trainer.test_dataloaders.dataset[idx]
            )
            look_back_window = look_back_window.unsqueeze(0).to(lightning_module.device)
            look_back_window_norm, mean, std = local_z_norm(
                look_back_window, lightning_module.local_norm_channels
            )
            target = target.unsqueeze(0)
            if lightning_module.has_probabilistic_forecast:
                pred_mean, pred_std = lightning_module.model_forward(
                    look_back_window_norm
                )
                pred_mean = pred_mean[:, :, : target.shape[-1]]
                pred_denorm = local_z_denorm(
                    pred_mean, lightning_module.local_norm_channels, mean, std
                )
                pred_std = pred_std[:, :, : target.shape[-1]] * std
            else:
                pred = lightning_module.model_forward(look_back_window_norm)[
                    :, :, : target.shape[-1]
                ]
                pred_denorm = local_z_denorm(
                    pred, lightning_module.local_norm_channels, mean, std
                )
                pred_std = None

            assert pred_denorm.shape == target.shape

            if hasattr(lightning_module.trainer.datamodule, "use_heart_rate"):
                use_heart_rate = lightning_module.trainer.datamodule.use_heart_rate
            else:
                use_heart_rate = False

            plot_prediction_wandb(
                look_back_window,
                target,
                pred_denorm,
                wandb_logger=lightning_module.logger,
                metric_name=metric_name,
                metric_value=v[idx],
                type=type,
                use_heart_rate=use_heart_rate,
                freq=lightning_module.trainer.datamodule.freq,
                dataset=lightning_module.trainer.datamodule.name,
                pred_denorm_std=pred_std,
            )


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
