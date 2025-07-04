import argparse
import numpy as np
import wandb
import torch
import plotly.io as pio
import plotly.graph_objects as go
import plotly.colors as pcolors
import seaborn as sns
import matplotlib.pyplot as plt

from tqdm import tqdm
from lightning.pytorch.loggers import WandbLogger
from plotly.subplots import make_subplots

from src.constants import (
    dataset_to_name,
)

from src.normalization import (
    local_z_denorm,
    local_z_norm,
    global_z_denorm,
    global_z_norm,
)


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


def plot_entire_series(
    logger: WandbLogger,
    datamodule,
    metrics: dict,
    plot_metrics: bool = True,
    data_type: str = "test",
    show_fig: bool = False,
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
    dataset = datamodule.name
    if data_type == "test":
        data = datamodule.test_dataset.data
    elif data_type == "train":
        data = datamodule.train_dataset.data
    elif data_type == "val":
        data = datamodule.val_dataset.data
    else:
        raise NotImplementedError()

    look_back_window = datamodule.look_back_window
    prediction_window = datamodule.prediction_window
    use_dynamic_features = datamodule.use_dynamic_features
    target_channel_dim = datamodule.target_channel_dim
    use_heart_rate = datamodule.use_heart_rate
    if plot_metrics:
        required_keys = {"MSE", "MAE", "dir_acc_single"}
        assert required_keys.issubset(metrics.keys())
        n_metrics = len(required_keys)  # only plot mse, mae and dir acc
    series_title = (
        ["Ground Truth Time Series", "Activity Info"]
        if use_dynamic_features
        else ["Ground Truth Time Series"]
    )
    subplot_titles = series_title
    if plot_metrics:
        subplot_titles += [
            name_to_title[metric_name]
            for metric_name in ["MSE", "MAE", "dir_acc_single"]
        ]

    colors = pcolors.qualitative.Plotly
    if plot_metrics:
        series_heights = [0.35, 0.35] if use_dynamic_features else [0.7]
        row_heights = series_heights + [0.3 / n_metrics for _ in range(n_metrics)]
    else:
        series_heights = [0.5, 0.5] if use_dynamic_features else [1.0]
        row_heights = series_heights

    window_length = look_back_window + prediction_window
    lengths = [len(s) - window_length + 1 for s in data]
    cum_lengths = np.cumsum([0] + lengths)

    if plot_metrics:
        assert cum_lengths[-1] == len(metrics["MSE"])

    max_series = 30
    if len(data) >= max_series:
        print(f"We have a lot of timeseries. Only plot the first {max_series}.")
        data = data[:max_series]
    n_series = len(data)
    for j in range(n_series):
        print(f"Plotting entire series {j + 1}")
        n_rows = 2 if use_dynamic_features else 1
        if plot_metrics:
            n_rows += n_metrics
        fig = make_subplots(
            rows=n_rows,
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
                mode="lines" if not use_heart_rate else None,
                name=f"{get_yaxis_name(dataset, use_heart_rate)} Value",
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
        if plot_metrics:
            offset = 1 if use_dynamic_features else 0
            for i, metric_name in enumerate(["MSE", "MAE", "dir_acc_single"]):
                mse_metric = np.array(
                    metrics[metric_name][cum_lengths[j] : cum_lengths[j + 1]]
                )

                mse_plot = np.zeros(n)
                assert len(mse_metric) < len(mse_plot)
                mse_plot[
                    look_back_window - 1 : look_back_window - 1 + len(mse_metric)
                ] = mse_metric

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

        heart_rate = "HR" if use_heart_rate else ""
        activity_info = "Activity" if use_dynamic_features else ""
        fig.update_layout(
            title_text=f"{dataset_to_name[dataset]} {heart_rate} {activity_info} LBW={look_back_window} PW={prediction_window}",
            title_x=0.5,  # centers the title
        )

        if show_fig:
            fig.show()
        else:
            logger.experiment.log(
                {f"entire_series/series_vs_metric_{j}": wandb.Html(pio.to_html(fig))}
            )


def plot_all_data(datamodule, data_type: str = "test", save_html: bool = False) -> None:
    dataset = datamodule.name
    if data_type == "test":
        data = datamodule.test_dataset.data
    elif data_type == "train":
        data = datamodule.train_dataset.data
    elif data_type == "val":
        data = datamodule.val_dataset.data
    else:
        raise NotImplementedError()

    use_dynamic_features = datamodule.use_dynamic_features
    use_static_features = datamodule.use_static_features
    target_channel_dim = datamodule.target_channel_dim
    use_heart_rate = datamodule.use_heart_rate

    colors = pcolors.qualitative.Plotly + pcolors.qualitative.Dark24
    max_series = 20
    if len(data) >= max_series:
        print(f"We have a lot of timeseries. Only plot the first {max_series}.")
        data = data[:max_series]
    n_series = len(data)
    n_rows = 2 if use_dynamic_features or use_static_features else 1

    row_titles = []
    for i in range(n_series):
        row_titles.append(f"{get_yaxis_name(dataset, use_heart_rate)}")
        if n_rows > 1:
            row_titles.append("Activity")

    fig = make_subplots(
        rows=n_rows * n_series,
        cols=1,
        shared_xaxes=False,
        row_titles=row_titles,
        # vertical_spacing=0.05,
        # row_heights=row_heights,
        # subplot_titles=subplot_titles,
    )
    for j in range(n_series):
        print(f"Plotting entire series {j + 1}")
        series = data[j][:, 0]
        activity = data[j][:, target_channel_dim]
        n = len(series)
        fig.add_trace(
            go.Scatter(
                x=list(range(n)),
                y=series,
                mode="lines" if not use_heart_rate else "markers",
                name=f"P{j + 1}",
                line=dict(color=colors[j]),
            ),
            row=n_rows * j + 1,
            col=1,
        )
        if use_dynamic_features:
            fig.add_trace(
                go.Scatter(
                    x=list(range(n)),
                    y=activity,
                    mode="lines",
                    name=f"P{j + 1} Activity Value",
                    line=dict(color=colors[j]),
                    showlegend=False,
                ),
                row=n_rows * j + 2,
                col=1,
            )
            fig.update_xaxes(matches=f"x{2 * j + 1}", row=2 * j + 2, col=1)
        elif use_static_features:

            def int_to_str(activity_label: int):
                if activity_label == 0:
                    return "Transition"
                elif activity_label == 1:
                    return "Sitting"
                elif activity_label == 2:
                    return "Ascending / Descending Stairs"
                elif activity_label == 3:
                    return "Table Soccer"
                elif activity_label == 4:
                    return "Cycling"
                elif activity_label == 5:
                    return "Driving Car"
                elif activity_label == 6:
                    return "Lunch Break"
                elif activity_label == 7:
                    return "Walking"
                elif activity_label == 8:
                    return "Working"

            # TODO: process activity and plot the activity string

            fig.add_trace(
                go.Scatter(
                    x=list(range(n)),
                    y=activity,
                    mode="lines",
                    name=f"P{j + 1} Activity Value",
                    line=dict(color=colors[j]),
                    showlegend=False,
                ),
                row=n_rows * j + 2,
                col=1,
            )

            fig.update_xaxes(matches=f"x{2 * j + 1}", row=2 * j + 2, col=1)

    fig.update_layout(
        title={
            "text": f"<b>{dataset_to_name[dataset]} Dataset </b>",
            "x": 0.5,
            "xanchor": "center",
            "font": dict(
                size=40, family="Arial", color="black"
            ),  # bold by default for many fonts
        },
        legend=dict(font=dict(size=16)),
        height=n_rows * 2000,
    )

    if save_html:
        activity_string = "Activity" if use_dynamic_features else ""
        plot_name = (
            f"{dataset} {get_yaxis_name(dataset, use_heart_rate)} {activity_string}"
        )
        pio.write_html(fig, file=f"./plots/data/{plot_name}.html", auto_open=True)
    else:
        fig.show()


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

    if use_heart_rate and dataset not in ["mhc6mwt"]:
        # dalia, wildppg and ieee use a sliding window with stride 2 seconds for the HR
        t_lookback = np.arange(-(len(look_back_window) - 1), 1) * 2
        t_future = np.arange(0, len(target)) * 2
    elif use_heart_rate and dataset in ["mhc6mwt"]:
        # the mhc6mwt dataset has frequency 1 for the heartrate values
        t_lookback = np.arange(-(len(look_back_window) - 1), 1)
        t_future = np.arange(0, len(target))
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
    normalization = lightning_module.normalization
    local_norm_channels = lightning_module.local_norm_channels

    test_dataset = lightning_module.trainer.datamodule.test_dataset

    metrics_to_plot = ["MSE", "MAE", "cross_correlation"]

    # plot best, worst and median
    for metric_name in metrics_to_plot:
        v = lightning_module.metric_full[metric_name]

        sorted_indices = np.argsort(v)
        min_idx = sorted_indices[0]
        max_idx = sorted_indices[-1]
        median_idx = sorted_indices[len(v) // 2]
        if metric_name in ["cross_correlation"]:
            order = [min_idx, max_idx, median_idx]
        else:
            order = [max_idx, min_idx, median_idx]

        for type, idx in zip(["worst", "best", "median"], order):
            look_back_window, target = test_dataset[idx]
            look_back_window = look_back_window.unsqueeze(0).to(lightning_module.device)

            # normalize the lookback window
            if normalization == "local":
                look_back_window_norm, mean, std = local_z_norm(
                    look_back_window, local_norm_channels
                )
            elif normalization == "global":
                datamodule = lightning_module.trainer.datamodule
                mean = datamodule.train_dataset.mean
                std = datamodule.train_dataset.std
                look_back_window_norm = global_z_norm(
                    look_back_window, local_norm_channels, mean, std
                )
            elif normalization == "none":
                look_back_window_norm = look_back_window

            target = target.unsqueeze(0)
            if lightning_module.has_probabilistic_forecast:
                pred_mean, pred_std = lightning_module.model_forward(
                    look_back_window_norm
                )
                pred_mean = pred_mean[:, :, : target.shape[-1]]
                pred_std = pred_std[:, :, : target.shape[-1]]
                if normalization in ["local", "global"]:
                    pred_std *= std
            else:
                pred_mean = lightning_module.model_forward(look_back_window_norm)[
                    :, :, : target.shape[-1]
                ]
                pred_std = None

            # denormalize the prediction
            if normalization == "local":
                pred_denorm = local_z_denorm(pred_mean, local_norm_channels, mean, std)
            elif normalization == "global":
                pred_denorm = global_z_denorm(pred_mean, local_norm_channels, mean, std)
            elif normalization == "none":
                pred_denorm = pred_mean

            assert pred_denorm.shape == target.shape, (
                f"target shape = {target.shape} | pred_denorm shape = {pred_denorm.shape}"
            )

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
    from hydra import initialize, compose
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from src.utils import (
        compute_square_window,
        compute_input_channel_dims,
        get_optuna_name,
    )

    OmegaConf.register_new_resolver("compute_square_window", compute_square_window)
    OmegaConf.register_new_resolver("eval", eval)
    OmegaConf.register_new_resolver("optuna_name", get_optuna_name)
    OmegaConf.register_new_resolver(
        "compute_input_channel_dims", compute_input_channel_dims
    )

    parser = argparse.ArgumentParser(description="WandB Results")

    parser.add_argument(
        "--dataset",
        type=str,
        choices=["wildppg", "dalia", "ieee", "mhc6mwt"],
        required=True,
        default="ieee",
        help="Dataset to plot. Must be 'ieee', 'dalia', 'wildppg' or 'mhc6mwt' ",
    )

    parser.add_argument(
        "--granger",
        required=False,
        action="store_true",
        help="checks granger causality for the timeseries in the dataset",
    )

    args = parser.parse_args()

    with initialize(version_base=None, config_path="../config/"):
        cfg = compose(
            config_name="config",
            overrides=[
                f"dataset={args.dataset}",
                "experiment=all",
                "use_dynamic_features=True",
            ],
        )

    datamodule = instantiate(cfg.dataset.datamodule)

    datamodule.setup("fit")
    # datamodule.setup("test")
    if args.granger:
        from statsmodels.tsa.stattools import grangercausalitytests

        lags = [1, 3, 5, 10, 20, 30]
        print(
            f"Computing Granger Causality for dataset {dataset_to_name[datamodule.name]} with lags {lags}"
        )
        dataset = datamodule.train_dataset.data

        fig = make_subplots(
            rows=len(dataset),
            cols=1,
            column_titles=[f"Series {i + 1}" for i in range(len(dataset))],
            shared_xaxes=False,
            vertical_spacing=0.05,
        )

        for i, series in tqdm(enumerate(dataset)):
            p_values = []
            for lag in lags:
                gc_res = grangercausalitytests(series, [lag], verbose=False)
                p_value = gc_res[lag][0]["ssr_ftest"][1]
                p_values.append(round(p_value, 3))
            fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=p_values,
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
            fig.update_xaxes(
                title_text="Lags",
                tickmode="array",
                tickvals=lags,
                row=i + 1,
                col=1,
            )

        fig.update_layout(
            title={
                "text": f"<b>Granger Causality P-Values for {dataset_to_name[datamodule.name]}</b>",
                "x": 0.5,
                "xanchor": "center",
                "font": dict(size=40, family="Arial", color="black"),
            },
        )

        fig.show()

    else:
        plot_all_data(datamodule, data_type="train", save_html=False)
